"""Process fine-tuning results into scored inventory summaries."""

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from experiment_config import (
    BFI10_ITEMS_TRAIT_POLARITY,
    IPIP_ITEMS_FILE,
    PANAS_X_TRAIT_SUBCLASSES,
)

DEBUG = True
VERBOSE = DEBUG


IPIP_ITEMS_DF = pd.read_csv(IPIP_ITEMS_FILE)
INVENTORY_ITEM_COUNTS = {
    "PANASX": 29,
    "BFI10": 10,
    "IPIP120": IPIP_ITEMS_DF.shape[0],
}
IPIP_POLARITY_LOOKUP = {
    row["phrase"].strip().lower(): row["polarity"].strip() for _, row in IPIP_ITEMS_DF.iterrows()
}
BFI10_POLARITY_LOOKUP = {
    entry["item"].strip(): entry["polarity"].strip() for entry in BFI10_ITEMS_TRAIT_POLARITY
}

TRAIT_PREFIX = "t_"
FACET_PREFIX = "f_"

ALL_TRAIT_COLUMNS = [
    f"{TRAIT_PREFIX}{t}" for t in ["O", "C", "E", "A", "N", "Anger", "Optimism", "Joy", "Sadness"]
]

IPIP_FACET_COLUMNS = sorted(IPIP_ITEMS_DF["facet"].dropna().unique())
ALL_FACET_COLUMNS = [f"{FACET_PREFIX}{f}" for f in IPIP_FACET_COLUMNS]

META_COLUMNS = [
    "dataset",
    "split",
    "split_trait",
    "split_location",
    "split_size",
    "seed",
    "use_peft",
    "lora_scale",
    "epoch",
    "step",
    "inventory",
    "item",
    "likert_in_prompt",
    "phase",
]


OCEAN_SHORT = ["O", "C", "E", "A", "N"]
TRAIT_COLS = [f"t_{t}" for t in OCEAN_SHORT] + ["t_Anger", "t_Optimism", "t_Joy", "t_Sadness"]

IPIP_FACETS = [
    "Anxiety",
    "Anger",
    "Depression",
    "Self-Consciousness",
    "Immoderation",
    "Vulnerability",
    "Friendliness",
    "Gregariousness",
    "Assertiveness",
    "Activity Level",
    "Excitement-Seeking",
    "Cheerfulness",
    "Trust",
    "Morality",
    "Altruism",
    "Cooperation",
    "Modesty",
    "Sympathy",
    "Self-Efficacy",
    "Orderliness",
    "Dutifulness",
    "Achievement-Striving",
    "Self-Discipline",
    "Cautiousness",
    "Imagination",
    "Artistic Interests",
    "Emotionality",
    "Adventurousness",
    "Intellect",
    "Liberalism",
]
FACET_COLS = [f"f_{f.replace(' ', '_')}" for f in IPIP_FACET_COLUMNS]

SCORE_COLUMNS = TRAIT_COLS + FACET_COLS


def parse_exp_id(df: pd.DataFrame) -> pd.DataFrame:
    """Parse experiment IDs into normalized metadata columns."""

    def parse_id(exp_id: str) -> dict[str, object | None]:
        dataset_map = {"E": "emotion", "P": "pandora"}
        parts = exp_id.split("-")
        if len(parts) < 5:
            return dict.fromkeys(
                ["dataset", "split", "split_trait", "split_location", "split_size", "seed"], None
            )
        dataset = dataset_map.get(parts[0], "unknown")
        seed = next((int(parts[i + 1]) for i in range(len(parts) - 1) if parts[i] == "Se"), None)

        if dataset == "emotion":
            emotion_splits = {"a": "anger", "h": "happiness", "j": "joy", "o": "optimism"}
            return {
                "dataset": dataset,
                "split": emotion_splits.get(parts[1], "unknown"),
                "split_trait": None,
                "split_location": None,
                "split_size": None,
                "seed": seed,
            }
        elif dataset == "pandora":
            trait_map = {
                "o": "openness",
                "c": "conscientiousness",
                "e": "extraversion",
                "a": "agreeableness",
                "n": "neuroticism",
            }
            location_map = {"t": "top", "b": "bottom"}
            trait_abbr, location_abbr = parts[1][0], parts[1][1]
            return {
                "dataset": dataset,
                "split": parts[1],
                "split_trait": trait_map.get(trait_abbr, "unknown"),
                "split_location": location_map.get(location_abbr, "unknown"),
                "split_size": int(parts[1][2:]),
                "seed": seed,
            }
        return dict.fromkeys(
            ["dataset", "split", "split_trait", "split_location", "split_size", "seed"], None
        )

    parsed = df["exp_id"].map(parse_id).apply(pd.Series)
    for c in parsed.columns:
        df[c] = parsed[c]
    return df


def add_likert_values(df: pd.DataFrame) -> pd.DataFrame:
    """Compute max-choice and weighted Likert summary columns."""
    likert_cols = list("12345")
    df[likert_cols] = df[likert_cols].astype(float)
    df["variant_likert_max"] = df[likert_cols].idxmax(axis=1).astype("int8")
    weights = np.arange(1, 6)
    df["variant_likert_weighted"] = (df[likert_cols] * weights).sum(axis=1).astype("float64")
    return df


def reverse_code_bfi10(df: pd.DataFrame, index_range: range) -> None:
    """Reverse-code negatively keyed BFI-10 items."""
    for idx in index_range:
        row = df.loc[idx]
        item = row["item"].strip()
        if BFI10_POLARITY_LOOKUP.get(item) == "negative":
            df.at[idx, "variant_likert_max"] = 6 - row["variant_likert_max"]
            df.at[idx, "variant_likert_weighted"] = 6 - row["variant_likert_weighted"]


def reverse_code_ipip120(df: pd.DataFrame, index_range: range) -> None:
    """Reverse-code negatively keyed IPIP-120 items."""
    for idx in index_range:
        row = df.loc[idx]
        item = row["item"].strip().lower()
        if IPIP_POLARITY_LOOKUP.get(item) == "-":
            df.at[idx, "variant_likert_max"] = 6 - row["variant_likert_max"]
            df.at[idx, "variant_likert_weighted"] = 6 - row["variant_likert_weighted"]


def _score_bfi10(batch: pd.DataFrame, variant_col: str) -> dict:
    """Score a BFI-10 batch into OCEAN trait means."""
    scores = {col: np.nan for col in SCORE_COLUMNS}
    by_trait = {t: [] for t in OCEAN_SHORT}
    for _, row in batch.iterrows():
        item = row["item"].strip()
        trait = next(
            e["trait"][0]  # first letter: O, C, …
            for e in BFI10_ITEMS_TRAIT_POLARITY
            if e["item"] == item
        )
        by_trait[trait].append(row[variant_col])
    for t, vals in by_trait.items():
        scores[f"t_{t}"] = float(np.mean(vals))
    return scores


_PANAS_MAP = {}
for lbl, words in PANAS_X_TRAIT_SUBCLASSES.items():
    for w in words:
        _PANAS_MAP[w.lower()] = lbl.title()


def _score_panasx(batch: pd.DataFrame, variant_col: str) -> dict:
    """Score a PANAS-X batch into emotion trait means."""
    scores = {col: np.nan for col in SCORE_COLUMNS}
    tmp = {"Anger": [], "Sadness": [], "Joy": [], "Optimism": []}
    for _, r in batch.iterrows():
        stem = r["item"].rstrip(".").lower()
        label = _PANAS_MAP.get(stem, None)
        if label:
            tmp[label].append(r[variant_col])
    for lbl, vals in tmp.items():
        scores[f"t_{lbl}"] = float(np.mean(vals)) if vals else np.nan
    return scores


_FACET_TO_ITEMS = (
    IPIP_ITEMS_DF.assign(phrase=lambda d: d["phrase"].str.strip().str.lower())
    .groupby("facet")["phrase"]
    .apply(list)
    .to_dict()
)
_FACET_TO_TRAIT = (
    IPIP_ITEMS_DF.groupby("facet")["trait"].first().apply(lambda x: x[0].upper()).to_dict()
)


def _score_ipip120(batch: pd.DataFrame, variant_col: str) -> dict:
    """Score an IPIP-120 batch into trait and facet means."""
    scores = {col: np.nan for col in SCORE_COLUMNS}
    facet_means = {}
    phrase_to_val = dict(zip(batch["item"].str.strip().str.lower(), batch[variant_col]))
    for facet, items in _FACET_TO_ITEMS.items():
        vals = [phrase_to_val[i] for i in items]
        facet_means[facet] = float(np.mean(vals))
        scores[f"f_{facet.replace(' ', '_')}"] = facet_means[facet]
    by_trait = {t: [] for t in OCEAN_SHORT}
    for fct, m in facet_means.items():
        by_trait[_FACET_TO_TRAIT[fct]].append(m)
    for t, vals in by_trait.items():
        scores[f"t_{t}"] = float(np.mean(vals))
    return scores


def _dispatch_score(batch: pd.DataFrame, variant_col: str) -> dict:
    """Dispatch batch scoring to the inventory-specific helper."""
    inv = batch.iloc[0]["inventory"]
    if inv == "BFI10":
        return _score_bfi10(batch, variant_col)
    if inv == "PANASX":
        return _score_panasx(batch, variant_col)
    if inv == "IPIP120":
        return _score_ipip120(batch, variant_col)
    raise ValueError(f"Unknown inventory {inv}")


def score_inventories(df_processed: pd.DataFrame) -> pd.DataFrame:
    """Score each inventory batch twice: max-choice and weighted Likert."""
    scored_rows: list[dict[str, object]] = []
    i = 0
    while i < len(df_processed):
        inv = df_processed.iloc[i]["inventory"]
        n_items = INVENTORY_ITEM_COUNTS[inv]
        idx = range(i, i + n_items)
        batch = df_processed.iloc[idx]
        meta = batch.iloc[0][META_COLUMNS].to_dict()

        for variant_col, tag in [
            ("variant_likert_max", "max"),
            ("variant_likert_weighted", "weighted"),
        ]:
            row = dict(meta)
            row["variant_likert"] = tag
            row.update(_dispatch_score(batch, variant_col))
            scored_rows.append(row)

        i += n_items

    scored_df = pd.DataFrame(scored_rows)
    col_order = META_COLUMNS + ["variant_likert"] + SCORE_COLUMNS
    for c in col_order:
        if c not in scored_df:
            scored_df[c] = np.nan
    return scored_df[col_order]


def score_and_save(df_after_reverse: pd.DataFrame, base_name: str, output_dir: str) -> None:
    """Score a reversed dataframe and persist the resulting CSV."""
    print(f"Step 4: Scoring inventories for {base_name}")
    scored = score_inventories(df_after_reverse)
    out_fp = os.path.join(output_dir, f"{base_name}_scored.csv")
    scored.to_csv(out_fp, index=False)
    print(f"Saved scored file to {out_fp}")


TRANSFORMATION_PIPELINE = {
    "BFI10": [reverse_code_bfi10],
    "IPIP120": [reverse_code_ipip120],
    "PANASX": [],
}


def process_inventory_batches(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse-code each inventory batch in sequence."""
    df = df.copy()
    i = 0
    total_batches = 0
    j = 0
    while j < len(df):
        inv = df.iloc[j]["inventory"]
        count = INVENTORY_ITEM_COUNTS.get(inv, 0)
        j += count
        total_batches += 1

    with tqdm(total=total_batches, desc="Step 3: Reverse coding batches") as pbar:
        while i < len(df):
            inv = df.iloc[i]["inventory"]
            if inv not in INVENTORY_ITEM_COUNTS:
                raise ValueError(f"Unknown inventory '{inv}' at row {i}")
            count = INVENTORY_ITEM_COUNTS[inv]
            indices = range(i, i + count)
            chunk = df.iloc[indices]
            if not all(chunk["inventory"] == inv):
                raise ValueError(f"Inconsistent inventory at row {i}")
            for fn in TRANSFORMATION_PIPELINE.get(inv, []):
                fn(df, indices)
            i += count
            pbar.update(1)
    return df


def run_pipeline(input_path: str, output_dir: str, drop_columns: list[str]) -> str:
    """Run the full processing pipeline for a single results file."""
    base_name = Path(input_path).stem
    df = pd.read_csv(input_path, dtype={"lora_scale": "object"})
    df = df.loc[:, ~df.columns.duplicated()]
    df.columns = df.columns.map(str)

    if VERBOSE:
        print(f"Step 1: Parsing exp_id for {base_name}")
    df = parse_exp_id(df)
    if DEBUG:
        df.to_csv(os.path.join(output_dir, f"{base_name}_1_parsed.csv"), index=False)

    for col in META_COLUMNS:
        if col not in df.columns:
            df[col] = np.nan

    if VERBOSE:
        print(f"Step 2: Adding likert scores for {base_name}")
    df = add_likert_values(df)
    if DEBUG:
        df.to_csv(os.path.join(output_dir, f"{base_name}_2_likert.csv"), index=False)

    if VERBOSE:
        print(f"Step 3: Reverse coding inventories for {base_name}")
    df = process_inventory_batches(df)
    if DEBUG:
        df.to_csv(os.path.join(output_dir, f"{base_name}_3_reversed.csv"), index=False)

    score_and_save(df, base_name, output_dir)
    return f"Finished processing {base_name}"


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    REPO_ROOT = Path(__file__).resolve().parents[1]
    OUTPUTS_DIR = REPO_ROOT / "outputs_best"
    RAW_DIR = OUTPUTS_DIR / "raw"
    DROP_COLUMNS = []
    FILES = [
        # RAW_DIR / 'combined_mid_epoch_results.csv',
        # RAW_DIR / 'combined_post_epoch_results.csv'
        RAW_DIR / "merged_results.csv"
    ]
    for fp in FILES:
        print(f"\nProcessing {fp}")
        result = run_pipeline(str(fp), str(OUTPUTS_DIR), DROP_COLUMNS)
        print(result)
