"""
Attempt 1

Since XGBDistribution already returns a Normal per row, we
wrap it in BoTorch's Model interface and then use the library's own acquisition
functions and batch optimiser


Usage

    python3 botorch_1_xgbdistv3.py --acq qlcb --q 10 --pool 200000
"""

import argparse
import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import torch

from botorch.acquisition import (
    qLowerConfidenceBound,
    qUpperConfidenceBound,
    qProbabilityOfImprovement,
    qLogExpectedImprovement,
)
from botorch.models.model import Model
from botorch.optim import optimize_acqf_discrete
from botorch.posteriors.gpytorch import GPyTorchPosterior
from gpytorch.distributions import MultivariateNormal
from scipy.stats import norm

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.double)


bundle_path = os.path.join(os.path.dirname(__file__), "/Users/PrathamShah/Desktop/Tubingen /Codes/XGBoost_Distribution/xgbd03_v3_bundle.json")
model_path = os.path.join(os.path.dirname(__file__), "/Users/PrathamShah/Desktop/Tubingen /Codes/XGBoost_Distribution/xgbd03_v3_model.joblib")

def load_bundle(bundle_path):
    """Read the JSON bundle and the model file it names.

    """
    with open(bundle_path) as f:
        spec = json.load(f)
    model_path = os.path.join(os.path.dirname(os.path.abspath(bundle_path)),
                              spec["model_file"])
    booster = joblib.load(model_path)

    print(f"Loaded {spec['model_family']} v{spec['version']}: {spec['version_label']}")
    print(f"  features    : {len(spec['feature_cols'])}")
    print(f"  sigma_scale : {spec['sigma_scale']:.4f} ({spec['calibration_source']})")
    print(f"  held-out R2 : {spec['metrics_heldout_broth']['r2']:.4f}   "
          f"RMSE {spec['metrics_heldout_broth']['rmse']:.4f}")
    return spec, booster


# wrapping in botorch interface

class XGBDistributionSurrogate(Model):
    """Expose the trained XGBDistribution model as a BoTorch `Model`.


    XGBDistribution gives us marginal means and standard deviations, one per
    row, and says nothing about how two candidates relate to each other. But a
    q-batch acquisition function needs the JOINT distribution over the q points,
    otherwise it has no way to know that two near-identical recipes carry
    near-identical information, and the batch fills up with duplicates.

    We therefore supply the correlation structure with an RBF kernel over the
    nutrient composition vectors, and keep the model's own marginal variances:

        Cov(x, x') = sigma(x) * k(x, x') * sigma(x'),
        k(x, x')   = exp(-||R0(x) - R0(x')||^2 / (2 * lengthscale^2))

    Two recipes with nearly the same nutrient vector are nearly perfectly
    correlated, so the second one adds almost nothing to the batch and the
    acquisition naturally spreads out. This is what replaces the ad hoc
    minimum-distance rule you would otherwise have to write yourself.

    Calibration
   
    Every sigma is multiplied by spec['sigma_scale'] before it becomes a
    posterior. The raw sigma is honest for interpolation but ~20 percent too
    small on unseen broths, which is the only regime the optimiser works in.
    """

    _num_outputs = 1

    def __init__(self, booster, spec, lengthscale=6.0):
        super().__init__()
        self.booster = booster
        self.spec = spec
        self.lengthscale = float(lengthscale)
        self.feature_cols = spec["feature_cols"]
        self.r0_idx = [self.feature_cols.index(c) for c in spec["r0_cols"]]
        self.sigma_scale = float(spec["sigma_scale"])
        self.y_lo, self.y_hi = spec["y_bounds"]

    @property
    def num_outputs(self):
        return 1

    def _predict_np(self, X_flat):
        """Call XGBDistribution on a 2-D numpy block and return calibrated mu, sd.

        The DataFrame is rebuilt with spec['feature_cols'] as the column names,
        which enforces the positional contract. Tree models index features by
        position, so a reordered input returns confident nonsense with no error.
        """
        df = pd.DataFrame(X_flat, columns=self.feature_cols)
        p = self.booster.predict(df)
        mu = np.clip(np.asarray(p.loc, dtype=float), self.y_lo, self.y_hi)
        sd = np.maximum(np.asarray(p.scale, dtype=float) * self.sigma_scale, 1e-6)
        return mu, sd

    def posterior(self, X, output_indices=None, observation_noise=False,
                  posterior_transform=None, **kwargs):
        """Return a joint Normal posterior over the q points in X.

        X arrives shaped `batch_shape x q x d`. It is flattened for the XGBoost
        call, then reshaped back so BoTorch's Monte Carlo samplers see the batch
        structure they expect.
        """
        q, d = X.shape[-2], X.shape[-1]
        batch_shape = X.shape[:-2]

        flat = X.reshape(-1, d).detach().cpu().numpy()
        mu, sd = self._predict_np(flat)

        mean = torch.as_tensor(mu).reshape(*batch_shape, q)
        std = torch.as_tensor(sd).reshape(*batch_shape, q)

        r0 = X[..., self.r0_idx]
        k = torch.exp(-torch.cdist(r0, r0) ** 2 / (2.0 * self.lengthscale ** 2))
        cov = std.unsqueeze(-1) * k * std.unsqueeze(-2)
        cov = cov + torch.eye(q, dtype=cov.dtype, device=cov.device) * 1e-6

        mvn = MultivariateNormal(mean, cov)
        posterior = GPyTorchPosterior(mvn)
        if posterior_transform is not None:
            posterior = posterior_transform(posterior)
        return posterior


def median_lengthscale(R0, rng, n=2000):
    """Median-heuristic lengthscale for the correlation kernel.

    The kernel needs a scale at which two recipes count as 'the same'. Taking
    the median pairwise distance in the candidate pool sets it from the data
    rather than by hand, and keeps it correct if the nutrient budget changes.
    """
    idx = rng.choice(len(R0), size=min(n, len(R0)), replace=False)
    d = torch.cdist(torch.as_tensor(R0[idx]), torch.as_tensor(R0[idx]))
    return float(torch.median(d[d > 0]))


# feasible set

def sample_feasible(spec, n, rng, alpha=1.0):
    """Draw `n` recipes obeying every constraint the bundle records.

    The constraint set is:
      * diversity level D in spec['diversity_levels']                  (categorical)
      * exactly D nutrients active out of 20                           (NChooseK)
      * total nutrient mass = spec['total_r0_mean']                    (mixture equality)
      * each active concentration inside spec['r0_bounds']             (box)
      * active flags and diversity_level derived, never sampled        (consistency)

    Concentrations are built by giving each chosen nutrient its observed floor
    and splitting the remaining mass with a Dirichlet draw. That satisfies the
    equality constraint exactly, instead of sampling and rejecting.

    Note on tooling: BoTorch's gradient-based `optimize_acqf` accepts linear
    equality constraints, but not NChooseK, and it cannot be used here anyway
    because a boosted tree has no gradient. Generating the choice set ourselves
    and calling `optimize_acqf_discrete` is the supported route for a
    non-differentiable surrogate. If the constraint set grows more complex,
    BoFire (built on BoTorch) has native NChooseK and mixture constraint
    handling and is the natural next step.
    """
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
        if m == 0:
            continue

        # A nutrient can only join a size-D recipe if its observed cap can hold
        # its share of the mass. At D = 1 this excludes c5, c13, c15, c17, c20,
        # which cap near 10.5 because they were never a sole carbon source.
        eligible = np.flatnonzero(hi >= total / D - TOL)
        if len(eligible) < D:
            continue

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
    if not good.all():
        print(f"  dropped {int((~good).sum()):,} infeasible draws")
        R0, D_col = R0[good], D_col[good]

    A = (R0 > 0).astype(float)
    cand = pd.DataFrame(R0, columns=r0_cols)
    for j, c in enumerate(active_cols):
        cand[c] = A[:, j]
    cand["diversity_level"] = D_col.astype(float)
    return cand[spec["feature_cols"]]


def dedupe(cand, spec, decimals=2):
    """Collapse candidates that are the same recipe to within rounding.

    `optimize_acqf_discrete(unique=True)` guarantees distinct INDICES, not
    distinct recipes. At D = 1 there are only fifteen admissible recipes in the
    whole space, so an unfiltered choice set contains thousands of copies of
    each and the batch comes back with the same broth listed twice. Rounding to
    two decimals of nutrient mass is well below anything a pipette or a MiCRM
    parameter sweep would distinguish.
    """
    key = cand[spec["r0_cols"]].round(decimals).astype(str).agg("|".join, axis=1)
    return cand.loc[~key.duplicated()].reset_index(drop=True)


def check_feasible(cand, spec, tol=1e-3):
    """Assert the pool obeys the constraints. Cheap, so always run it.

    An infeasible row is not a bad suggestion, it is a point with no training
    support at all, where the model's sigma carries no information.
    """
    r0 = cand[spec["r0_cols"]].values
    act = cand[spec["active_cols"]].values
    dlv = cand["diversity_level"].values
    assert np.allclose(r0.sum(axis=1), spec["total_r0_mean"], atol=tol), "mass not conserved"
    assert np.array_equal(act, (r0 > 0).astype(float)), "flags disagree with concentrations"
    assert np.array_equal(dlv, act.sum(axis=1)), "diversity_level disagrees with flag count"
    assert np.isin(dlv, spec["diversity_levels"]).all(), "unobserved diversity level"


def mutate(cand, spec, rng, n_mut=6, move_frac=0.30):
    """Enrich the choice set with local neighbours of the screened candidates.

    A boosted tree is piecewise constant, so there is no gradient for BoTorch's
    continuous optimiser to follow. Local search therefore has to be done by
    perturbation. Two mass-conserving moves are used: transfer part of one
    active nutrient's mass to another active one, or move one active nutrient's
    whole mass to a currently inactive one. Diversity level is never mutated,
    so each level keeps its own representatives in the choice set.
    """
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


# ----------------------------------------------------------------------------
# 4. Acquisition functions, straight from BoTorch
# ----------------------------------------------------------------------------

def build_acquisition(kind, model, spec, beta=2.0, threshold=0.95):
    """Instantiate the requested BoTorch q-batch acquisition function.

    qlcb  qLowerConfidenceBound. The PRESCRIPTION objective. Sigma here is
          largely the spread of suppression across the different communities
          grown in one broth, so maximising the lower bound asks for a recipe
          that suppresses whichever community happens to colonise it.
    qucb  qUpperConfidenceBound. The EXPLORATION objective, for deciding which
          recipes are worth spending a MiCRM run on: it favours places where
          the model might be wrong in our favour.
    qpi   qProbabilityOfImprovement with best_f = threshold. P(y > 0.95), the
          closest match to an eradication goal, which is about reliably
          clearing a bar rather than a high average.
    qlogei  qLogExpectedImprovement over the best observed y. Included for
          completeness. It is close to useless here: y_bounds already tops out
          at 0.99948, so the incumbent sits at the ceiling and there is almost
          no improvement left to expect.

    Note on beta: BoTorch's q-UCB is E[max(mu + sqrt(beta*pi/2)*|z|)], so beta
    is not the kappa of a textbook mu + kappa*sigma. beta = 2.0 corresponds to
    roughly 1.77 standard deviations.
    """
    if kind == "qlcb":
        return qLowerConfidenceBound(model, beta=beta)
    if kind == "qucb":
        return qUpperConfidenceBound(model, beta=beta)
    if kind == "qpi":
        return qProbabilityOfImprovement(model, best_f=threshold)
    if kind == "qlogei":
        return qLogExpectedImprovement(model, best_f=spec["y_bounds"][1])
    raise ValueError(f"unknown acquisition: {kind}")


def marginal_screen(cand, model, spec, kind, beta, threshold, keep_per_level):
    """Rank the pool by the cheap single-point version of the acquisition.

    `optimize_acqf_discrete` scores every choice at every greedy step, so
    handing it 200,000 rows means millions of surrogate calls. Screening first
    on the marginal (single-point) score and passing only the survivors is the
    standard way to keep the discrete optimiser tractable, and it costs nothing
    in quality because a candidate that is poor on its own will never be picked
    in a batch either. Screening is done per diversity level so no level is
    eliminated before BoTorch has seen it.
    """
    mu, sd = model._predict_np(cand.values)
    if kind == "qlcb":
        s = mu - np.sqrt(beta * np.pi / 2.0) * sd
    elif kind == "qucb":
        s = mu + np.sqrt(beta * np.pi / 2.0) * sd
    elif kind == "qpi":
        s = norm.cdf((mu - threshold) / sd)
    else:
        z = (mu - spec["y_bounds"][1]) / sd
        s = (mu - spec["y_bounds"][1]) * norm.cdf(z) + sd * norm.pdf(z)

    dl = cand["diversity_level"].values
    keep = []
    for D in spec["diversity_levels"]:
        idx = np.flatnonzero(dl == D)
        if len(idx) == 0:
            continue
        keep.append(idx[np.argsort(s[idx])[::-1][:keep_per_level]])
    keep = np.concatenate(keep)
    return cand.iloc[keep].reset_index(drop=True), mu, sd


# ----------------------------------------------------------------------------
# 5. Reporting
# ----------------------------------------------------------------------------

def describe(X_batch, model, spec, acq_values):
    """Turn the chosen tensor batch into a readable and re-runnable table."""
    r0_cols = spec["r0_cols"]
    arr = X_batch.detach().cpu().numpy()
    mu, sd = model._predict_np(arr)
    z = norm.ppf(0.975)
    r0 = arr[:, [spec["feature_cols"].index(c) for c in r0_cols]]

    rows = []
    for i in range(len(arr)):
        act = np.flatnonzero(r0[i] > 0)
        rows.append({
            "rank": i + 1,
            "diversity_level": int(round(arr[i, spec["feature_cols"].index("diversity_level")])),
            "recipe": "  ".join(f"{r0_cols[j].replace('R0_', '')}:{r0[i, j]:.2f}" for j in act),
            "pred_mean": mu[i],
            "pred_sigma": sd[i],
            "lo95": max(mu[i] - z * sd[i], spec["y_bounds"][0]),
            "hi95": min(mu[i] + z * sd[i], spec["y_bounds"][1]),
            "p_above_0.95": norm.cdf((mu[i] - 0.95) / sd[i]),
            "acq_value": float(acq_values[i]) if acq_values is not None else np.nan,
        })
    return pd.DataFrame(rows), arr


# ----------------------------------------------------------------------------
# 6. Main
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", default="xgbd03_v3_bundle.json")
    ap.add_argument("--acq", default="qlcb", choices=["qlcb", "qucb", "qpi", "qlogei"])
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--pool", type=int, default=200_000)
    ap.add_argument("--q", type=int, default=10, help="batch size to propose")
    ap.add_argument("--keep-per-level", type=int, default=600)
    ap.add_argument("--mutate-rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-per-level", action="store_true",
                    help="optimise one joint q-batch instead of one per diversity level")
    ap.add_argument("--out", default="bo_botorch_round1_recipes.csv")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    spec, booster = load_bundle(args.bundle)

    print(f"\nSampling {args.pool:,} feasible recipes")
    pool = sample_feasible(spec, args.pool, rng)
    check_feasible(pool, spec)

    r0_pool = pool[spec["r0_cols"]].values
    ls = median_lengthscale(r0_pool, rng)
    model = XGBDistributionSurrogate(booster, spec, lengthscale=ls)
    print(f"  kernel lengthscale (median heuristic): {ls:.2f} mass units")

    mu0, sd0 = model._predict_np(pool.values)
    print(f"  pool mean mu = {mu0.mean():.4f}   best mu = {mu0.max():.4f}   "
          f"mean sigma = {sd0.mean():.4f}")

    # Screen, then enrich the survivors with local mutations, re-screening each
    # time so the choice set handed to BoTorch is both small and good.
    choices, _, _ = marginal_screen(dedupe(pool, spec), model, spec, args.acq,
                                    args.beta, args.threshold, args.keep_per_level)
    for r in range(1, args.mutate_rounds + 1):
        children = mutate(choices, spec, rng)
        merged = pd.concat([choices, children], ignore_index=True)
        choices, _, _ = marginal_screen(dedupe(merged, spec), model, spec, args.acq,
                                        args.beta, args.threshold, args.keep_per_level)
        m_r, s_r = model._predict_np(choices.values)
        print(f"  local search round {r}: choice set {len(choices):,}, "
              f"best mu {m_r.max():.4f}")
    check_feasible(choices, spec)

    print(f"\nOptimising {args.acq} (beta={args.beta}) over {len(choices):,} "
          f"discrete choices, q={args.q}")
    acq = build_acquisition(args.acq, model, spec, args.beta, args.threshold)

    if args.no_per_level:
        # One joint q-batch over the whole choice set. The correlated posterior
        # already stops the batch filling with near-duplicates, so this is the
        # right mode once the interesting diversity level is known.
        X_batch, acq_vals = optimize_acqf_discrete(
            acq_function=acq, q=args.q,
            choices=torch.as_tensor(choices.values),
            max_batch_size=1024, unique=True)
        X_batch = X_batch.reshape(-1, len(spec["feature_cols"]))
        acq_vals = acq_vals.reshape(-1)
    else:
        # Round 1 spends its runs to LEARN the response surface, so the batch is
        # forced to span diversity levels: a separate q-batch is optimised
        # within each level. A single joint batch would land almost entirely on
        # whichever level currently scores best and would tell us nothing about
        # whether diversity matters.
        levels = spec["diversity_levels"]
        per = max(1, args.q // len(levels))
        parts, vals = [], []
        dl = choices["diversity_level"].values
        for D in levels:
            sub = choices[dl == D]
            if len(sub) == 0:
                continue
            xb, av = optimize_acqf_discrete(
                acq_function=acq, q=min(per, len(sub)),
                choices=torch.as_tensor(sub.values),
                max_batch_size=1024, unique=True)
            # With q = 1 BoTorch returns a scalar acquisition value and a 1-D
            # point, so both are reshaped before they are concatenated.
            xb = xb.reshape(-1, len(spec["feature_cols"]))
            av = av.reshape(-1)
            print(f"  D={D:<3} picked {len(xb)}   best acq {float(av.max()):.4f}")
            parts.append(xb)
            vals.append(av)
        X_batch = torch.cat(parts, dim=0)
        acq_vals = torch.cat(vals, dim=0)

    batch, arr = describe(X_batch, model, spec, acq_vals)

    pd.set_option("display.width", 220)
    print(f"\nProposed batch ({args.acq}):")
    print(batch[["rank", "diversity_level", "pred_mean", "pred_sigma", "lo95",
                 "hi95", "p_above_0.95"]].to_string(
                     index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nRecipes:")
    for _, r in batch.iterrows():
        print(f"  #{int(r['rank']):>2}  D={int(r['diversity_level']):<3} {r['recipe']}")

    print("\nBest predicted mean per diversity level (sampled pool):")
    dl = pool["diversity_level"].values
    for D in spec["diversity_levels"]:
        m = dl == D
        if m.sum():
            j = np.flatnonzero(m)[np.argmax(mu0[m])]
            print(f"  D={D:<3} best mu={mu0[j]:.4f}  sigma={sd0[j]:.4f}  n={m.sum():,}")

    out = pd.DataFrame(arr, columns=spec["feature_cols"])
    for c in ["rank", "recipe", "pred_mean", "pred_sigma", "lo95", "hi95",
              "p_above_0.95", "acq_value"]:
        out[c] = batch[c].values
    out["acquisition"] = args.acq
    out["model_version"] = spec["version"]
    out.to_csv(args.out, index=False)
    print(f"\nWrote {args.out} ({len(out)} recipes, full feature columns included)")


if __name__ == "__main__":
    main()