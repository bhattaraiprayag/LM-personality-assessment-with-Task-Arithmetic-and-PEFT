### Usage:
#   python decorrelate_ocean.py \
#       --scores   outputs_best/merged_results_scored.csv \
#       --truth    data/ocean_splits_info.csv \
#       --out      outputs_best/merged_results_decor.csv \
#       --ridge
#
import argparse, pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, Ridge

TRAITS = ["O", "C", "E", "A", "N"]
T_COLS = [f"t_{t}" for t in TRAITS]

def load_truth(truth_fp: str) -> pd.DataFrame:
    """Return truth with columns ['split_trait','split_location','split_size','t_O',…]."""
    truth = pd.read_csv(truth_fp, dtype={"size": "Int64"})
    # 1) rename cols to match scored csv
    truth = truth.rename(columns={
        "location": "split_location",
        "size":     "split_size",
        **{f"{t}_mean": f"t_{t}" for t in TRAITS}
    })
    # 2) keep only the three keys + 5 means
    keep = ["split_trait", "split_location", "split_size"] + T_COLS
    return truth[keep]

def fit_betas(df: pd.DataFrame, ridge=False):
    """Return 5×5 β matrix where β[j,k] is leakage from trait k → model score j."""
    X = df[[f"t_{t}" for t in TRAITS]].to_numpy(dtype=float)
    betas = np.zeros((5,5))
    Regr = Ridge if ridge else LinearRegression
    for j, target in enumerate(TRAITS):
        y = df[f"t_{target}"].to_numpy(dtype=float)
        # drop rows with nan in y
        mask = ~np.isnan(y)
        if mask.sum() < 5:            # not enough data
            betas[j] = np.nan
            continue
        reg = Regr(fit_intercept=True).fit(X[mask], y[mask])
        betas[j] = reg.coef_
    return betas

def apply_residualisation(df, betas):
    """Overwrite t_* columns with debiased scores."""
    X = df[T_COLS].to_numpy(float)
    # Y_adj_j = Y_j - sum_{k≠j} β_{j,k} X_k
    leakage = X @ (betas.T - np.diag(np.diag(betas)))   # (n×5)·(5×5)
    Y_adj = X - leakage
    df.loc[:, T_COLS] = Y_adj
    return df

def main(args):
    scored = pd.read_csv(args.scores, dtype={"split_size": "Int64"})
    truth  = load_truth(args.truth)

    # -------- restrict to the rows we actually debias --------
    mask = (scored["dataset"] == "pandora") & scored["inventory"].isin(["BFI10","IPIP120"])
    foc   = scored.loc[mask].copy()

    # -------- merge ground-truth means --------
    foc = foc.merge(truth, on=["split_trait","split_location","split_size"],
                    suffixes=("", "_true"), how="left", validate="m:1")

    # -------- residualise (one β matrix for all pandora rows) --------
    betas = fit_betas(foc[[c+"_true" for c in T_COLS] + T_COLS]
                         .rename(columns={f"{c}_true": c for c in T_COLS}), ridge=args.ridge)

    foc = apply_residualisation(foc, betas)

    ### FACETS — to be filled in later #################################
    # Same pattern but with 30-dim X; keep betas_facets separately.
    ###################################################################

    # -------- put debiased rows back & save --------
    scored.loc[mask, T_COLS] = foc[T_COLS]
    scored.to_csv(args.out, index=False)
    print(f"Saved debiased file →  {args.out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--truth",  required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--ridge",  action="store_true",
                    help="use ridge regression (α=1.0) instead of OLS")
    main(ap.parse_args())