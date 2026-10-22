"""
BoTorch optimisation with the XGBDistribution surrogate entering through the
library's own EnsembleModel / EnsemblePosterior interface, following the
suggested tutorial.

the surrogate is an ensemble of independently trained XGBDistributionn 
members, so the posterior carries epistemic (between-member) uncertainty
screening uses the acquisition function itself at q = 1
intervals and exceedance probabilities are computed from the mixture
    of member distributions, and the uncertainty is reported split into its
    epistemic and aleatoric parts

Usage

    python3 botorch_xgbd_ensembles.py --bundle train_xgbd_ensembles_output/xgbd03_v3e_bundle.json --acq qlcb --q 20
"""


import argparse
import json
import os
import warnings

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

_THREADS = os.environ.get("BO_NUM_THREADS", "1")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.special import ndtri
from scipy.stats import norm

from botorch.acquisition import (
    qLogExpectedImprovement,
    qLowerConfidenceBound,
    qProbabilityOfImprovement,
    qUpperConfidenceBound,
)
from botorch.models.ensemble import EnsembleModel
from botorch.optim import optimize_acqf_discrete
from botorch.sampling.index_sampler import IndexSampler

warnings.filterwarnings("ignore", message=".*serialized model.*")
torch.set_default_dtype(torch.double)
torch.set_num_threads(int(_THREADS))

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

# loading

def load_ensemble(bundle_path):
    """Read the ensemble bundle and every member it names, pinned to the thread cap."""
    with open(bundle_path) as f:
        spec = json.load(f)
    root = os.path.dirname(os.path.abspath(bundle_path))
    members = [joblib.load(os.path.join(root, name)) for name in spec["model_files"]]
    for m in members:
        m.get_booster().set_param({"nthread": int(_THREADS)})
        assert m.get_booster().num_features() == len(spec["feature_cols"]), \
            "member feature count does not match the bundle"

    print(f"Loaded {spec['model_family']}: {len(members)} members")
    print(f"  sigma_scale : {spec['sigma_scale']:.4f} ({spec['calibration_source']})")
    return spec, members


# surrogate (botorch EnsembleModel) and its posterior (EnsemblePosterior)

def _splitmix64(x):
    x = x + _GOLDEN
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


class XGBDistributionEnsemble(EnsembleModel):
    """K XGBDistribution members exposed through BoTorch's EnsembleModel.

    `forward` returns a `batch_shape x s x q x 1` tensor, which EnsembleModel's
    default `posterior()` wraps in an EnsemblePosterior. Two modes:

    epistemic  s = K. Each member contributes its mean. The spread is pure
               model disagreement.
    total      s = K * J. Each member contributes J draws from its own
               calibrated Normal, mu_k + sigma_k * eps_j. The mixture then holds
               both disagreement between members and community-to-community
               noise within each member.
    """

    def __init__(self, members, spec, mode="total", n_noise=16, seed=0):
        super().__init__()
        self._num_outputs = 1
        self.members = members
        self.feature_cols = spec["feature_cols"]
        self.r0_idx = [self.feature_cols.index(c) for c in spec["r0_cols"]]
        self.sigma_scale = float(spec["sigma_scale"])
        self.y_lo, self.y_hi = spec["y_bounds"]
        self.mode = mode
        self.n_noise = n_noise
        self.seed = np.uint64(seed)

    def member_predictions(self, X_flat):
        """Calibrated (mu, sigma) from every member, each shaped K x n."""
        df = pd.DataFrame(X_flat, columns=self.feature_cols)
        preds = [m.predict(df) for m in self.members]
        mu = np.stack([np.asarray(p.loc, dtype=float) for p in preds])
        sd = np.stack([np.asarray(p.scale, dtype=float) for p in preds]) * self.sigma_scale
        return mu, np.maximum(sd, 1e-6)

    def recipe_noise(self, X_flat):
        """J standard-normal draws per recipe, identical every time that recipe is seen.

        The draws are a deterministic hash of the rounded nutrient vector, so a
        recipe keeps the same noise across every call the optimiser makes
        (common random numbers), while different recipes get independent noise.
        """
        r0 = pd.DataFrame(np.round(X_flat[:, self.r0_idx], 4))
        keys = pd.util.hash_pandas_object(r0, index=False).values.astype(np.uint64)
        j = np.arange(self.n_noise, dtype=np.uint64)[:, None]
        z = _splitmix64(keys[None, :] ^ _splitmix64(j + self.seed))
        u = ((z >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)
        return ndtri(u)

    def forward(self, X):
        lead = X.shape[:-1]
        flat = X.reshape(-1, X.shape[-1]).detach().cpu().numpy()
        mu, sd = self.member_predictions(flat)
        if self.mode == "epistemic":
            values = mu
        else:
            eps = self.recipe_noise(flat)
            values = (mu[:, None, :] + sd[:, None, :] * eps[None, :, :]).reshape(-1, flat.shape[0])
        values = np.clip(values, self.y_lo, self.y_hi)
        out = torch.as_tensor(values, dtype=X.dtype).reshape(-1, *lead)
        return out.movedim(0, -2).unsqueeze(-1)


# feasible set

def sample_feasible(spec, n, rng, alpha=1.0):
    """Draw n recipes obeying the diversity, NChooseK, mixture and box constraints."""
    TOL = 1e-3
    r0_cols, active_cols = spec["r0_cols"], spec["active_cols"]
    K = len(r0_cols)
    total = float(spec["total_r0_mean"])
    lo = np.array([spec["r0_bounds"][c][0] for c in r0_cols])
    hi = np.array([spec["r0_bounds"][c][1] for c in r0_cols])
    levels = spec["diversity_levels"]

    R0 = np.zeros((n, K))
    D_col = np.zeros(n, dtype=int)
    for D, idx in zip(levels, np.array_split(np.arange(n), len(levels))):
        m = len(idx)
        eligible = np.flatnonzero(hi >= total / D - TOL)
        support = eligible[np.argsort(rng.random((m, len(eligible))), axis=1)[:, :D]]
        floors = lo[support]
        budget = np.maximum(total - floors.sum(axis=1), 0.0)
        g = rng.gamma(shape=alpha, scale=1.0, size=(m, D))
        conc = floors + budget[:, None] * (g / g.sum(axis=1, keepdims=True))

        caps = hi[support] + TOL
        excess = np.maximum(conc - caps, 0.0).sum(axis=1)
        conc = np.minimum(conc, caps)
        room = np.maximum(caps - conc, 0.0)
        rs = room.sum(axis=1)
        ok = rs > 1e-9
        conc[ok] += room[ok] * (excess[ok] / rs[ok])[:, None]

        block = np.zeros((m, K))
        np.put_along_axis(block, support, conc, axis=1)
        R0[idx] = block
        D_col[idx] = D

    good = np.abs(R0.sum(axis=1) - total) < TOL
    R0, D_col = R0[good], D_col[good]
    A = (R0 > 0).astype(float)
    cand = pd.DataFrame(R0, columns=r0_cols)
    for j, c in enumerate(active_cols):
        cand[c] = A[:, j]
    cand["diversity_level"] = D_col.astype(float)
    return cand[spec["feature_cols"]]


def dedupe(cand, spec, decimals=2):
    """Keep one row per recipe, identity judged on nutrient mass to two decimals."""
    arr = np.round(cand[spec["r0_cols"]].values, decimals)
    _, first = np.unique(arr, axis=0, return_index=True)
    return cand.iloc[np.sort(first)].reset_index(drop=True)


def check_feasible(cand, spec, tol=1e-3):
    """Assert every constraint holds for every row."""
    r0 = cand[spec["r0_cols"]].values
    act = cand[spec["active_cols"]].values
    dlv = cand["diversity_level"].values
    assert np.allclose(r0.sum(axis=1), spec["total_r0_mean"], atol=tol), "mass not conserved"
    assert np.array_equal(act, (r0 > 0).astype(float)), "flags disagree with concentrations"
    assert np.array_equal(dlv, act.sum(axis=1)), "diversity_level disagrees with flag count"
    assert np.isin(dlv, spec["diversity_levels"]).all(), "unobserved diversity level"


def mutate(cand, spec, rng, n_mut=6, move_frac=0.30):
    """Mass-conserving local moves: transfer between active nutrients, or swap one out."""
    r0_cols = spec["r0_cols"]
    lo = np.array([spec["r0_bounds"][c][0] for c in r0_cols])
    hi = np.array([spec["r0_bounds"][c][1] for c in r0_cols])
    out = np.repeat(cand[r0_cols].values, n_mut, axis=0)
    D_rep = np.repeat(cand["diversity_level"].values, n_mut)

    for row in out:
        act = np.flatnonzero(row > 0)
        inact = np.flatnonzero(row == 0)
        if len(act) < 2 or (len(inact) and rng.random() < 0.35):
            if not len(inact):
                continue
            src, dst = act[rng.integers(len(act))], inact[rng.integers(len(inact))]
            if row[src] > hi[dst]:
                continue
            row[dst], row[src] = row[src], 0.0
        else:
            src, dst = rng.choice(act, size=2, replace=False)
            delta = row[src] * rng.uniform(0.05, move_frac)
            if row[src] - delta < lo[src] or row[dst] + delta > hi[dst]:
                continue
            row[src] -= delta
            row[dst] += delta

    mut = pd.DataFrame(out, columns=r0_cols)
    A = (out > 0).astype(float)
    for j, c in enumerate(spec["active_cols"]):
        mut[c] = A[:, j]
    mut["diversity_level"] = D_rep
    return mut[spec["feature_cols"]]

# acquisition and screening

def build_acquisition(kind, model, spec, beta, threshold, n_mc, seed):
    """BoTorch q-acquisition with a seeded IndexSampler, so scores are reproducible."""
    sampler = IndexSampler(sample_shape=torch.Size([n_mc]), seed=seed)
    if kind == "qlcb":
        return qLowerConfidenceBound(model, beta=beta, sampler=sampler)
    if kind == "qucb":
        return qUpperConfidenceBound(model, beta=beta, sampler=sampler)
    if kind == "qpi":
        return qProbabilityOfImprovement(model, best_f=threshold, sampler=sampler)
    return qLogExpectedImprovement(model, best_f=spec["y_bounds"][1], sampler=sampler)


def score_single(acq, cand, chunk=4096):
    """The acquisition itself evaluated at q = 1 for every row of cand."""
    X = torch.as_tensor(cand.values).unsqueeze(-2)
    with torch.no_grad():
        return torch.cat([acq(X[i:i + chunk]) for i in range(0, len(X), chunk)]).numpy()


def screen(cand, acq, spec, keep_per_level):
    """Keep the top rows of each diversity level by single-point acquisition value."""
    s = score_single(acq, cand)
    dl = cand["diversity_level"].values
    keep = []
    for D in spec["diversity_levels"]:
        idx = np.flatnonzero(dl == D)
        keep.append(idx[np.argsort(s[idx])[::-1][:keep_per_level]])
    return cand.iloc[np.concatenate(keep)].reset_index(drop=True)


# summary

def mixture_summary(model, X_arr, threshold, spec):
    """Mean, uncertainty split, bounded 95 percent interval and P(y > threshold).

    The predictive law is the equal-weight mixture of the K calibrated member
    Normals. Its CDF is evaluated on a grid and inverted, then clipped to the
    physical bounds of suppression.
    """
    mu, sd = model.member_predictions(X_arr)
    mean = mu.mean(axis=0)
    epi = mu.std(axis=0)
    ale = np.sqrt((sd ** 2).mean(axis=0))
    grid = np.linspace(-0.5, 1.5, 4001)
    cdf = norm.cdf((grid[None, None, :] - mu[:, :, None]) / sd[:, :, None]).mean(axis=0)
    lo_q = grid[np.argmax(cdf >= 0.025, axis=1)]
    hi_q = grid[np.argmax(cdf >= 0.975, axis=1)]
    y_lo, y_hi = spec["y_bounds"]
    return {
        "pred_mean": np.clip(mean, y_lo, y_hi),
        "sigma_total": np.sqrt(epi ** 2 + ale ** 2),
        "sigma_epistemic": epi,
        "sigma_aleatoric": ale,
        "lo95": np.clip(lo_q, y_lo, y_hi),
        "hi95": np.clip(hi_q, y_lo, y_hi),
        f"p_above_{threshold}": (1 - norm.cdf((threshold - mu) / sd)).mean(axis=0),
    }


def recipe_string(row, r0_cols):
    return "  ".join(f"{c.replace('R0_', '')}:{row[i]:.2f}"
                     for i, c in enumerate(r0_cols) if row[i] > 0)


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="xgbd03_v3e_bundle.json")
    ap.add_argument("--acq", default="qlcb", choices=["qlcb", "qucb", "qpi", "qlogei"])
    ap.add_argument("--mode", default="total", choices=["total", "epistemic"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--pool", type=int, default=200_000)
    ap.add_argument("--q", type=int, default=10)
    ap.add_argument("--keep-per-level", type=int, default=600)
    ap.add_argument("--mutate-rounds", type=int, default=3)
    ap.add_argument("--n-noise", type=int, default=16)
    ap.add_argument("--n-mc", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-level", action="store_true")
    ap.add_argument("--out", default="bo_botorch_round2_recipes.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    spec, members = load_ensemble(args.bundle)
    model = XGBDistributionEnsemble(members, spec, mode=args.mode,
                                    n_noise=args.n_noise, seed=args.seed)
    acq = build_acquisition(args.acq, model, spec, args.beta, args.threshold,
                            args.n_mc, args.seed)
    s_size = len(members) * (args.n_noise if args.mode == "total" else 1)
    print(f"  mode        : {args.mode} (ensemble posterior size s = {s_size})", flush=True)

    print(f"\nSampling {args.pool:,} feasible recipes", flush=True)
    pool = dedupe(sample_feasible(spec, args.pool, rng), spec)
    check_feasible(pool, spec)
    print(f"  {len(pool):,} unique recipes after de-duplication", flush=True)

    choices = screen(pool, acq, spec, args.keep_per_level)
    for r in range(1, args.mutate_rounds + 1):
        merged = dedupe(pd.concat([choices, mutate(choices, spec, rng)], ignore_index=True), spec)
        choices = screen(merged, acq, spec, args.keep_per_level)
        print(f"  local search round {r}: choice set {len(choices):,}", flush=True)
    check_feasible(choices, spec)

    print(f"\nOptimising {args.acq} over {len(choices):,} choices, q={args.q}", flush=True)
    levels = [None] if args.no_per_level else spec["diversity_levels"]
    per = args.q if args.no_per_level else max(1, args.q // len(levels))
    batches = []
    for D in levels:
        sub = choices if D is None else choices[choices["diversity_level"] == D]
        xb, av = optimize_acqf_discrete(acq_function=acq, q=min(per, len(sub)),
                                        choices=torch.as_tensor(sub.values),
                                        max_batch_size=1024, unique=True)
        xb = xb.reshape(-1, len(spec["feature_cols"])).numpy()
        batches.append((D, xb, float(av.reshape(-1)[-1])))
        if D is not None:
            print(f"  D={D:<3} picked {len(xb)}   joint acq value {batches[-1][2]:.4f}", flush=True)

    X_arr = np.vstack([b[1] for b in batches])
    summ = mixture_summary(model, X_arr, args.threshold, spec)
    out = pd.DataFrame(X_arr, columns=spec["feature_cols"])
    out.insert(0, "recipe", [recipe_string(r, spec["r0_cols"])
                             for r in out[spec["r0_cols"]].values])
    for k, v in summ.items():
        out[k] = v
    out["level_batch_acq"] = np.concatenate([[b[2]] * len(b[1]) for b in batches])
    out = out.sort_values(["diversity_level", "pred_mean"], ascending=[True, False])
    out.insert(0, "rank_in_level", out.groupby("diversity_level").cumcount() + 1)

    for k, v in {"acquisition": args.acq, "mode": args.mode, "beta": args.beta,
                 "threshold": args.threshold, "pool": args.pool, "seed": args.seed,
                 "n_noise": args.n_noise, "n_mc": args.n_mc,
                 "ensemble_size": len(members),
                 "keep_per_level": args.keep_per_level}.items():
        out[k] = v

    pd.set_option("display.width", 220)
    show = ["diversity_level", "rank_in_level", "pred_mean", "sigma_total",
            "sigma_epistemic", "sigma_aleatoric", "lo95", "hi95", f"p_above_{args.threshold}"]
    print(f"\nProposed batch ({args.acq}, {args.mode}):")
    print(out[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nRecipes:")
    for _, r in out.iterrows():
        print(f"  D={int(r['diversity_level']):<3} #{int(r['rank_in_level'])}  {r['recipe']}")

    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} recipes, {out.shape[1]} columns)")


if __name__ == "__main__":
    main()