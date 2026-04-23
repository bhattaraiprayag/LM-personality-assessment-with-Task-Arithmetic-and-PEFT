"""Residualize OCEAN scores against ground-truth split statistics."""

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

TRAITS = ["O", "C", "E", "A", "N"]
T_COLS = [f"t_{t}" for t in TRAITS]


def load_truth(truth_fp: str) -> pd.DataFrame:
    """Load and normalize the ground-truth split statistics."""
    truth = pd.read_csv(truth_fp, dtype={"size": "Int64"})
    truth = truth.rename(
        columns={
            "location": "split_location",
            "size": "split_size",
            **{f"{t}_mean": f"t_{t}" for t in TRAITS},
        }
    )
    keep = ["split_trait", "split_location", "split_size"] + T_COLS
    return truth[keep]


def fit_betas(df: pd.DataFrame, ridge: bool = False) -> np.ndarray:
    """Fit a 5x5 leakage matrix from ground-truth trait scores."""
    X = df[[f"t_{t}" for t in TRAITS]].to_numpy(dtype=float)
    betas = np.zeros((5, 5))
    Regr = Ridge if ridge else LinearRegression
    for j, target in enumerate(TRAITS):
        y = df[f"t_{target}"].to_numpy(dtype=float)
        mask = ~np.isnan(y)
        if mask.sum() < 5:
            betas[j] = np.nan
            continue
        reg = Regr(fit_intercept=True).fit(X[mask], y[mask])
        betas[j] = reg.coef_
    return betas


def apply_residualisation(df: pd.DataFrame, betas: np.ndarray) -> pd.DataFrame:
    """Overwrite trait columns with residualized scores."""
    X = df[T_COLS].to_numpy(float)
    leakage = X @ (betas.T - np.diag(np.diag(betas)))
    Y_adj = X - leakage
    df.loc[:, T_COLS] = Y_adj
    return df


def main(args: argparse.Namespace) -> None:
    """Run the residualization pipeline and write the debiased CSV."""
    scored = pd.read_csv(args.scores, dtype={"split_size": "Int64"})
    truth = load_truth(args.truth)

    mask = (scored["dataset"] == "pandora") & scored["inventory"].isin(["BFI10", "IPIP120"])
    foc = scored.loc[mask].copy()

    foc = foc.merge(
        truth,
        on=["split_trait", "split_location", "split_size"],
        suffixes=("", "_true"),
        how="left",
        validate="m:1",
    )

    betas = fit_betas(
        foc[[c + "_true" for c in T_COLS] + T_COLS].rename(
            columns={f"{c}_true": c for c in T_COLS}
        ),
        ridge=args.ridge,
    )

    foc = apply_residualisation(foc, betas)
    scored.loc[mask, T_COLS] = foc[T_COLS]
    scored.to_csv(args.out, index=False)
    print(f"Saved debiased file →  {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ridge", action="store_true", help="use ridge regression (α=1.0) instead of OLS"
    )
    main(ap.parse_args())
