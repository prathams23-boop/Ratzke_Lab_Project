"""
Train an ensemble of XGBDistribution members for BoTorch's EnsembleModel.

Each member is an independent XGBDistribution fitted on a random 80 percent of
the broths (subbagging over broth_id), with the same hyperparameters as the V3
model. The held-out 20 percent of broths gives every member honest
out-of-broth predictions, which are pooled to recalibrate sigma.

Outputs, written next to the V3 bundle:
    <prefix>_member00.joblib ... <prefix>_memberKK.joblib
    <prefix>_bundle.json

Usage

    python3 train_xgbd_ensembles.py --data community_index_final.csv \
        --bundle xgbd03_v3_bundle.json --members 100
"""

import argparse
import json
import os

_THREADS = os.environ.get("BO_NUM_THREADS", "1")
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
             "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import GroupShuffleSplit
from xgboost_distribution import XGBDistribution

TARGET = "community_suppression"
GROUP = "broth_id"
EPS = 1e-6


def load_data(csv_path, spec):
    """Return the feature matrix in bundle order, the target and the broth groups."""
    df = pd.read_csv(csv_path)
    return df[spec["feature_cols"]], df[TARGET], df[GROUP].values


def inner_holdout(X, y, groups, frac, seed):
    """Group-aware split of a member's training broths for early stopping."""
    gss = GroupShuffleSplit(n_splits=1, test_size=frac, random_state=seed)
    fit_idx, val_idx = next(gss.split(X, y, groups=groups))
    return X.iloc[fit_idx], y.iloc[fit_idx], X.iloc[val_idx], y.iloc[val_idx]


def fit_member(X, y, groups, params, seed):
    """Fit one XGBDistribution member with the V3 hyperparameters and early stopping."""
    Xf, yf, Xv, yv = inner_holdout(X, y, groups, frac=0.15, seed=seed)
    model = XGBDistribution(
        distribution="normal",
        natural_gradient=True,
        n_estimators=1000,
        max_depth=int(params["max_depth"]),
        learning_rate=float(params["learning_rate"]),
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_weight=3,
        reg_lambda=1.0,
        random_state=seed,
        n_jobs=int(_THREADS),
        early_stopping_rounds=40,
        verbosity=0,
    )
    model.fit(Xf, yf, eval_set=[(Xv, yv)], verbose=False)
    return model


def calibration_factor(y, mu, sd, level=0.95):
    """Scalar that makes mu +/- z * s * sd reach `level` empirical coverage."""
    z_abs = np.abs(y - mu) / np.maximum(sd, EPS)
    return float(np.quantile(z_abs, level) / norm.ppf(0.5 + level / 2.0))


def scorecard(y, mu, sd):
    """R2, RMSE, Gaussian NLL and 95 percent coverage for one set of predictions."""
    sd = np.maximum(sd, EPS)
    z = norm.ppf(0.975)
    return {
        "r2": float(1 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2)),
        "rmse": float(np.sqrt(np.mean((y - mu) ** 2))),
        "nll": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + (y - mu) ** 2 / (2 * sd ** 2))),
        "cov95": float(np.mean(np.abs(y - mu) <= z * sd)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--bundle", default="xgbd03_v3_bundle.json")
    ap.add_argument("--members", type=int, default=10)
    ap.add_argument("--holdout", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--prefix", default="xgbd03_v3e")
    args = ap.parse_args()

    with open(args.bundle) as f:
        spec = json.load(f)
    out_dir = os.path.dirname(os.path.abspath(args.bundle))
    X, y, groups = load_data(args.data, spec)
    print(f"Data: {len(X):,} rows, {len(np.unique(groups))} broths, "
          f"{len(spec['feature_cols'])} features", flush=True)

    gss = GroupShuffleSplit(n_splits=args.members, test_size=args.holdout,
                            random_state=args.seed)
    files, member_metrics = [], []
    y_all, mu_all, sd_all = [], [], []

    for k, (tr, te) in enumerate(gss.split(X, y, groups=groups)):
        model = fit_member(X.iloc[tr], y.iloc[tr], groups[tr], spec["best_params"],
                           seed=args.seed + k)
        p = model.predict(X.iloc[te])
        yk, mu, sd = y.iloc[te].values, np.asarray(p.loc), np.asarray(p.scale)
        card = scorecard(yk, mu, sd)
        card.update({"member": k, "n_trees": int(model.best_iteration) + 1,
                     "n_heldout_broths": int(len(np.unique(groups[te])))})
        member_metrics.append(card)

        name = f"{args.prefix}_member{k:02d}.joblib"
        joblib.dump(model, os.path.join(out_dir, name))
        files.append(name)
        y_all.append(yk); mu_all.append(mu); sd_all.append(sd)
        print(f"  member {k:02d}: trees={card['n_trees']:<4} R2={card['r2']:.4f} "
              f"RMSE={card['rmse']:.4f} cov95(raw)={card['cov95']:.3f}", flush=True)

    y_all, mu_all, sd_all = map(np.concatenate, (y_all, mu_all, sd_all))
    sigma_scale = calibration_factor(y_all, mu_all, sd_all, 0.95)
    pooled_raw = scorecard(y_all, mu_all, sd_all)
    pooled_cal = scorecard(y_all, mu_all, sd_all * sigma_scale)

    bundle = {k: v for k, v in spec.items() if k != "model_file"}
    bundle.update({
        "model_family": "xgboost_distribution_ensemble",
        "version_label": spec["version_label"] + " -- broth-subbagged ensemble",
        "model_files": files,
        "ensemble_size": len(files),
        "member_holdout_frac": args.holdout,
        "ensemble_seed": args.seed,
        "sigma_scale": sigma_scale,
        "calibration_source": "pooled_member_heldout_broths",
        "metrics_member_pooled_raw": pooled_raw,
        "metrics_member_pooled_calibrated": pooled_cal,
        "member_metrics": member_metrics,
    })
    bundle_path = os.path.join(out_dir, f"{args.prefix}_bundle.json")
    with open(bundle_path, "w") as f:
        json.dump(bundle, f, indent=2)

    print(f"\nPooled held-out (members): R2={pooled_raw['r2']:.4f}  "
          f"RMSE={pooled_raw['rmse']:.4f}")
    print(f"sigma_scale = {sigma_scale:.4f}   cov95 raw {pooled_raw['cov95']:.3f} "
          f"-> calibrated {pooled_cal['cov95']:.3f}")
    print(f"Wrote {bundle_path} and {len(files)} member files")


if __name__ == "__main__":
    main()