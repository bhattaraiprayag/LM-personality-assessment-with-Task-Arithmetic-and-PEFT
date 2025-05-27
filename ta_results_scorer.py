import pandas as pd
import numpy as np
from pathlib import Path
from experiment_config import (
    BFI10_ITEMS_TRAIT_POLARITY,
    PANAS_X_TRAIT_SUBCLASSES,
    IPIP_ITEMS_FILE,
)

# -------- constants --------
IPIP_ITEMS_DF = pd.read_csv(IPIP_ITEMS_FILE)
INVENTORY_ITEM_COUNTS = {
    "PANASX": 29,
    "BFI10": 10,
    "IPIP120": IPIP_ITEMS_DF.shape[0],
}
IPIP_POLARITY_LOOKUP = {
    r["phrase"].strip().lower(): r["polarity"].strip() for _, r in IPIP_ITEMS_DF.iterrows()
}
BFI10_POLARITY_LOOKUP = {
    e["item"].strip(): e["polarity"].strip() for e in BFI10_ITEMS_TRAIT_POLARITY
}

TRAIT_PREFIX = "t_"
FACET_PREFIX = "f_"

OCEAN_SHORT = ["O", "C", "E", "A", "N"]
TRAIT_COLS = [f"{TRAIT_PREFIX}{t}" for t in OCEAN_SHORT] + [
    f"{TRAIT_PREFIX}{x}" for x in ["Anger", "Optimism", "Joy", "Sadness"]
]

IPIP_FACET_COLUMNS = sorted(IPIP_ITEMS_DF["facet"].dropna().unique())
FACET_COLS = [f"{FACET_PREFIX}{f.replace(' ', '_')}" for f in IPIP_FACET_COLUMNS]
SCORE_COLUMNS = TRAIT_COLS + FACET_COLS

META_COLUMNS = ["base-model", "lora-model", "lora_scale", "inventory", "likert_in_prompt"]

# -------- helpers --------
def add_likert_values(df):
    likert_cols = list("12345")
    df[likert_cols] = df[likert_cols].astype(float)
    df["variant_likert_max"] = df[likert_cols].idxmax(axis=1).astype(int)
    weights = np.arange(1, 6)
    df["variant_likert_weighted"] = (df[likert_cols] * weights).sum(axis=1)
    return df


def reverse_code(row):
    inv = row["inventory"]
    if inv == "BFI10":
        if BFI10_POLARITY_LOOKUP.get(row["item"].strip()) == "negative":
            row["variant_likert_max"] = 6 - row["variant_likert_max"]
            row["variant_likert_weighted"] = 6 - row["variant_likert_weighted"]
    elif inv == "IPIP120":
        if IPIP_POLARITY_LOOKUP.get(row["item"].strip().lower()) == "-":
            row["variant_likert_max"] = 6 - row["variant_likert_max"]
            row["variant_likert_weighted"] = 6 - row["variant_likert_weighted"]
    return row


def _score_bfi10(batch, col):
    out = {c: np.nan for c in SCORE_COLUMNS}
    by_trait = {t: [] for t in OCEAN_SHORT}
    for _, r in batch.iterrows():
        trait = next(e["trait"][0] for e in BFI10_ITEMS_TRAIT_POLARITY if e["item"] == r["item"].strip())
        by_trait[trait].append(r[col])
    for t, vals in by_trait.items():
        out[f"t_{t}"] = float(np.mean(vals))
    return out


_PANAS_MAP = {}
for k, wds in PANAS_X_TRAIT_SUBCLASSES.items():
    for w in wds:
        _PANAS_MAP[w.lower()] = k.title()


def _score_panasx(batch, col):
    out = {c: np.nan for c in SCORE_COLUMNS}
    tmp = {"Anger": [], "Sadness": [], "Joy": [], "Optimism": []}
    for _, r in batch.iterrows():
        stem = r["item"].rstrip(".").lower()
        lbl = _PANAS_MAP.get(stem)
        if lbl:
            tmp[lbl].append(r[col])
    for lbl, vals in tmp.items():
        if vals:
            out[f"t_{lbl}"] = float(np.mean(vals))
    return out


_FACET_TO_ITEMS = (
    IPIP_ITEMS_DF.assign(phrase=lambda d: d["phrase"].str.strip().str.lower())
    .groupby("facet")["phrase"]
    .apply(list)
    .to_dict()
)
_FACET_TO_TRAIT = (
    IPIP_ITEMS_DF.groupby("facet")["trait"].first().apply(lambda x: x[0].upper()).to_dict()
)


def _score_ipip120(batch, col):
    out = {c: np.nan for c in SCORE_COLUMNS}
    facet_means = {}
    p2v = dict(zip(batch["item"].str.strip().str.lower(), batch[col]))
    for facet, items in _FACET_TO_ITEMS.items():
        vals = [p2v[i] for i in items]
        m = float(np.mean(vals))
        facet_means[facet] = m
        out[f"f_{facet.replace(' ', '_')}"] = m
    by_trait = {t: [] for t in OCEAN_SHORT}
    for fct, m in facet_means.items():
        by_trait[_FACET_TO_TRAIT[fct]].append(m)
    for t, vals in by_trait.items():
        out[f"t_{t}"] = float(np.mean(vals))
    return out


def _dispatch(inv, batch, col):
    if inv == "BFI10":
        return _score_bfi10(batch, col)
    if inv == "PANASX":
        return _score_panasx(batch, col)
    if inv == "IPIP120":
        return _score_ipip120(batch, col)
    raise ValueError(inv)


def score_inventories(df):
    scored = []
    group_cols = META_COLUMNS
    for _, batch in df.groupby(group_cols, sort=False):
        inv = batch.iloc[0]["inventory"]
        if len(batch) != INVENTORY_ITEM_COUNTS[inv]:
            continue
        meta = batch.iloc[0][group_cols].to_dict()
        for col, tag in [("variant_likert_max", "max"), ("variant_likert_weighted", "weighted")]:
            row = dict(meta)
            row["variant_likert"] = tag
            row.update(_dispatch(inv, batch, col))
            scored.append(row)
    out = pd.DataFrame(scored)
    col_order = META_COLUMNS + ["variant_likert"] + SCORE_COLUMNS
    for c in col_order:
        if c not in out:
            out[c] = np.nan
    return out[col_order]


# -------- main --------
if __name__ == "__main__":
    INPUT_FILE = Path("ta_results.csv")
    OUTPUT_FILE = Path("ta_results_scored.csv")
    df = pd.read_csv(INPUT_FILE, dtype={"lora_scale": "object"})
    df = df.loc[:, ~df.columns.duplicated()]
    df = add_likert_values(df)
    df = df.apply(reverse_code, axis=1)
    scored = score_inventories(df)
    scored.to_csv(OUTPUT_FILE, index=False)
