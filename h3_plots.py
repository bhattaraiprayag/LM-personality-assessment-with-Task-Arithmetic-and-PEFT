import argparse, re, json, os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 600

OCEAN_LETTERS = ["O", "C", "E", "A", "N"]
LETTER2TRAIT = dict(zip(OCEAN_LETTERS, ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]))
EMOTION_TRAITS = ["Anger", "Sadness", "Joy", "Optimism"]
EMOTION_LETTER = {"a": "Anger", "s": "Sadness", "j": "Joy", "o": "Optimism"}
SPLIT_SIZES = [5, 10, 15]
LOC_MAP = {"t": "top", "b": "bottom"}
TICKS = [-25, -10, -5, -1, 0, 1, 5, 10, 25]

VARIANT_LIKERT_SCORING = "weighted"
VARIANT_LIKERT_INCLUSION = "include"
variant_suffix = f"{VARIANT_LIKERT_SCORING[0]}_{VARIANT_LIKERT_INCLUSION[0]}"


def style_symlog(ax):
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


def mean_sd(df: pd.DataFrame, col: str) -> pd.DataFrame:
    g = df.groupby("lora_scale")[col]
    return g.mean().to_frame("mean").join(g.std().to_frame("sd")).reset_index()


def trait2col(trait: str) -> str:
    t = trait.lower()
    if t in {v.lower() for v in LETTER2TRAIT.values()}:
        return f"t_{t[0].upper()}"
    return f"t_{trait.title()}"


def parse_model(mid: str) -> dict:
    # try:
    #     mid_l = mid.lower()
    # except AttributeError:
    #     print(f"Error parsing model ID: {mid}")
    # if mid_l.startswith("p-"):
    #     m = re.match(r"^p-([ocean])([tb])(\d+)", mid_l)
    #     # print(m)
    #     if not m:
    #         m = re.match(r"^p-([a-z])([tb])(\d+)", mid_l)
    #     letter, loc, size = m.groups()
    #     trait = LETTER2TRAIT[letter.upper()]
    #     return dict(
    #         dataset="pandora",
    #         trait=trait,
    #         location=LOC_MAP[loc],
    #         size=int(size),
    #         desc_full=f"{trait.lower()}-{LOC_MAP[loc]}-{size}",
    #         desc_short=trait.lower(),
    #     )
    """Return a dict describing the model ID or 'unknown' if it cannot be parsed."""
    default = dict(
        dataset="unknown", trait="unknown", location="unknown",
        size=None, desc_full=str(mid), desc_short=str(mid)
    )
    if not isinstance(mid, str) or pd.isna(mid):
        return default
    mid_l = mid.lower()
    if mid_l.startswith("p-"):
        m = re.match(r"^p-([ocean])([tb])(\d+)", mid_l) or \
            re.match(r"^p-([a-z])([tb])(\d+)", mid_l)
        if not m:
            print(f"Warning: Could not parse model ID '{mid}'")
            return default
        letter, loc, size = m.groups()
        trait = LETTER2TRAIT[letter.upper()]
        return dict(
            dataset="pandora",
            trait=trait,
            location=LOC_MAP[loc],
            size=int(size),
            desc_full=f"{trait.lower()}-{LOC_MAP[loc]}-{size}",
            desc_short=trait.lower(),
        )
    if mid_l.startswith("e-"):
        m = re.match(r"^e-([ajso])", mid_l)
        if not m:
            return default
        trait = EMOTION_LETTER[m.group(1)]
        return dict(
            dataset="emotion",
            trait=trait,
            location="normal",
            size=None,
            desc_full=trait.lower(),
            desc_short=trait.lower(),
        )
    # return dict(dataset="unknown", trait="unknown", location="unknown", size=None, desc_full=mid, desc_short=mid)
    return default


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lora_scale"] = df["lora_scale"].replace("baseline", 0).astype(float)
    df = df[(df["likert_in_prompt"] == VARIANT_LIKERT_INCLUSION) & (df["variant_likert"] == VARIANT_LIKERT_SCORING)]
    # print("Rows with NaN in base-model or lora-model:")
    # print(df[df["base-model"].isna() | df["lora-model"].isna()])
    # if df["base-model"].isna().any() or df["lora-model"].isna().any():
    #     print("Warning: Found missing model IDs:")
    #     print(df[df["base-model"].isna() | df["lora-model"].isna()])
    base_info = df["base-model"].apply(parse_model).apply(pd.Series).add_prefix("b_")
    lora_info = df["lora-model"].apply(parse_model).apply(pd.Series).add_prefix("l_")
    return pd.concat([df.reset_index(drop=True), base_info, lora_info], axis=1)


def plot_h3_1(df: pd.DataFrame, outdir: Path):
    for (bm, lm), g in df.groupby(["base-model", "lora-model"]):
        b = parse_model(bm)
        l = parse_model(lm)
        title_part = f"Base: {b['desc_full']} + LoRA: {l['desc_full']}"
        for inv, sub in g.groupby("inventory"):
            if inv == "PANASX":
                traits = [f"t_{t}" for t in EMOTION_TRAITS]
            else:
                traits = [f"t_{x}" for x in OCEAN_LETTERS]
            sub = sub.groupby("lora_scale")[traits].mean().reset_index()
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for col in traits:
                if col not in sub.columns or sub[col].isna().all():
                    continue
                ms = mean_sd(g, col)
                ax.plot(ms["lora_scale"], ms["mean"], marker="o", label=col[2:])
                ax.fill_between(ms["lora_scale"], ms["mean"] - ms["sd"], ms["mean"] + ms["sd"], alpha=0.15)
            style_symlog(ax)
            ax.set_ylabel("Score")
            ax.set_xlabel("LoRA scale")
            ax.set_title(f"{title_part} | {inv} scores")
            ax.legend()
            fname = f"{variant_suffix}_h3.1_{inv}_{b['desc_short']}_{l['desc_short']}.png"
            fig.tight_layout()
            fig.savefig(outdir / fname, bbox_inches="tight")
            plt.close(fig)


def gather_variable(df: pd.DataFrame, side: str, trait: str, loc: str, const_id: str) -> pd.DataFrame:
    if side == "b":
        f = (
            (df["b_trait"].str.lower() == trait.lower())
            & (df["b_location"] == loc)
            & (df["lora-model"] == const_id)
        )
    else:
        f = (
            (df["l_trait"].str.lower() == trait.lower())
            & (df["l_location"] == loc)
            & (df["base-model"] == const_id)
        )
    return df[f]



def is_pandora(info: dict) -> bool:
    """True if the parsed model came from the Pandora data-set."""
    return info["dataset"] == "pandora"


def fmt_pandora(info: dict) -> str:
    """
    Short label for a Pandora split: “agreeableness-top”, “openness-bottom”, …
    For non-Pandora models just return `desc_short` (anger, joy, …).
    """
    if is_pandora(info):
        return f"{info['trait'].lower()}-{info['location']}"
    return info["desc_short"]

def plot_h3_2(df: pd.DataFrame, outdir: Path):
    # for _, row in df[["base-model", "lora-model"]].drop_duplicates().iterrows():
    id_pairs = (
        df[["base-model", "lora-model"]]
        .dropna(subset=["base-model", "lora-model"])
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    for bm, lm in id_pairs:
        # bm, lm = row["base-model"], row["lora-model"]
        # print(f"Processing {bm} + {lm} ...")
        bi, li = parse_model(bm), parse_model(lm)
        # print(f"Parsed base model: {bi}, LoRA model: {li}")
        if is_pandora(bi) and not is_pandora(li):
            var_side, var_info, const_id, const_info = "b", bi, lm, li
        elif is_pandora(li) and not is_pandora(bi):
            var_side, var_info, const_id, const_info = "l", li, bm, bi
        else:
            print(f"Skipping non-Pandora models: {bi['dataset']} + {li['dataset']}")
            continue
        # if bi["dataset"] == li["dataset"]:
        #     print(f"Skipping same dataset models: {bi['dataset']}")
        #     continue
        # var_side, var_info, const_id, const_info = (
        #     ("b", bi, lm, li) if bi["dataset"] == "pandora" else ("l", li, bm, bi)
        # )
        var_side, var_info, const_id, const_info = (
            ("b", bi, lm, li) if bi["dataset"] == "pandora" else ("l", li, bm, bi)
        )
        var_trait = var_info["trait"]
        var_loc = var_info["location"]
        const_trait = const_info["trait"]
        sub = gather_variable(df, var_side, var_trait, var_loc, const_id)
        if sub.empty:
            continue
        inv_groups = sub.groupby("inventory")
        for inv, g in inv_groups:
            sizes = sorted([s for s in g[f"{var_side}_size"].dropna().unique() if s in SPLIT_SIZES])
            if not sizes:
                continue
            for tr in [var_trait, const_trait]:
                tcol = trait2col(tr)
                fig, ax = plt.subplots(figsize=(8, 6))
                for s in sizes:
                    gg = g[g[f"{var_side}_size"] == s]
                    if tcol not in gg.columns or gg[tcol].isna().all():
                        continue
                    ms = mean_sd(gg, tcol)
                    ax.plot(ms["lora_scale"], ms["mean"], marker="o", label=f"{s}%")
                    ax.fill_between(ms["lora_scale"], ms["mean"] - ms["sd"], ms["mean"] + ms["sd"], alpha=0.15)
                if ax.lines:
                    style_symlog(ax)
                    ax.set_ylabel("Score")
                    ax.set_xlabel("LoRA scale")
                    # base_desc = bi["desc_full"] if bi["dataset"] == "pandora" else bi["desc_full"]
                    base_desc = fmt_pandora(bi)
                    lora_desc = fmt_pandora(li)
                    # lora_desc = li["desc_short"] if li["dataset"] == "pandora" else li["desc_full"]
                    # title = f"Base: {base_desc} + LoRA: {lora_desc} | {inv} {tr} scores across {var_trait.lower()} splits"
                    title = (
                        f"Base: {base_desc} + LoRA: {lora_desc} | "
                        f"{inv} {tr} scores across {var_trait.lower()} splits"
                    )
                    ax.set_title(title)
                    ax.legend(title="Split size")
                    fname = (
                        f"{variant_suffix}_h3.2_{inv}_{base_desc}_{lora_desc}_{tr.lower()}.png"
                    )
                    fig.tight_layout()
                    fig.savefig(outdir / fname, bbox_inches="tight")
                plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scores", default="ta_results_scored.csv")
    p.add_argument("--outdir", default="viz_h3.1")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df = preprocess(pd.read_csv(args.scores))
    plot_h3_1(df, outdir)
    plot_h3_2(df, outdir)


if __name__ == "__main__":
    main()
