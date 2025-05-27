import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["figure.dpi"] = 600

OCEAN_LETTERS = ["O", "C", "E", "A", "N"]
OCEAN_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
LETTER2NAME = dict(zip(OCEAN_LETTERS, OCEAN_NAMES))
LETTER2LOWER = dict(zip(OCEAN_LETTERS, [n.lower() for n in OCEAN_NAMES]))
EMOTION_TRAITS = ["Anger", "Sadness", "Joy", "Optimism"]
EMOTION_COLS = [f"t_{t}" for t in EMOTION_TRAITS]
SPLIT_SIZES = [5, 10, 15]
LOCATIONS = ["top", "bottom"]
TICKS = [-25, -10, -5, -1, 0, 1, 5, 10, 25]
VARIANT_LIKERT_SCORING = "max"
VARIANT_LIKERT_INCLUSION = "exclude"
variant_suffix = f"{VARIANT_LIKERT_SCORING[0]}_{VARIANT_LIKERT_INCLUSION[0]}"


def mean_sd(df: pd.DataFrame, value: str, by: list[str]) -> pd.DataFrame:
    g = df.groupby(by)[value]
    m = g.mean().reset_index(name="mean")
    s = g.std().reset_index(name="sd")
    return m.merge(s, on=by)


def preprocess_baseline(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[(df["likert_in_prompt"] == VARIANT_LIKERT_INCLUSION) & (df["variant_likert"] == VARIANT_LIKERT_SCORING)]
    trait_cols = [c for c in df.columns if c.startswith("t_")]
    df[trait_cols] = df[trait_cols].apply(pd.to_numeric, errors="coerce")
    return df


def preprocess_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["lora_scale"] = df["lora_scale"].replace("baseline", 0.0).astype(float)
    df.loc[(df["dataset"] == "emotion") & (df["split"] == "unknown"), "split"] = "sadness"
    df = df[(df["likert_in_prompt"] == "include") & (df["variant_likert"] == "weighted")]
    return df


def plot_personality_h1(b_base: pd.DataFrame, res: pd.DataFrame, inv: str, outdir: Path):
    base_row = b_base[b_base["inventory"] == inv]
    if base_row.empty:
        return
    for letter in OCEAN_LETTERS:
        trait_lower = LETTER2LOWER[letter]
        base_val = base_row[f"t_{letter}"].mean()
        for loc in LOCATIONS:
            sub = res[
                (res["dataset"] == "pandora")
                & (res["inventory"] == inv)
                & (res["use_peft"] == "baseline")
                & (res["phase"] == "post")
                & (res["lora_scale"] == 0)          # new – keep pure baseline
                & (res["split_trait"] == trait_lower)
                & (res["split_location"] == loc)
                & (res["split_size"].isin(SPLIT_SIZES))
            ]
            if sub.empty:
                continue
            sub = sub.copy()
            sub.loc[:, "split_size"] = sub["split_size"].astype(float)
            agg = mean_sd(sub, f"t_{letter}", ["split_size"]).set_index("split_size")
            means = [base_val] + [agg.loc[s, "mean"] if s in agg.index else np.nan for s in SPLIT_SIZES]
            sds = [0.0] + [agg.loc[s, "sd"] if s in agg.index else 0.0 for s in SPLIT_SIZES]
            if all(np.isnan(means[1:])):
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            x = np.arange(4)
            colors = ["red"] + ["green"] * 3
            ax.bar(x, means, yerr=sds, capsize=5, color=colors, width=0.6)
            ax.set_xticks(x)
            ax.set_xticklabels(["Pretrained", "5", "10", "15"])
            ax.set_ylabel("Score")
            title = f"{LETTER2NAME[letter]} | Pre-trained vs. Finetuned ({loc.title()} splits) – {inv}"
            ax.set_title(title)
            fig.tight_layout()
            fname = f"{variant_suffix}_H1_{inv}_{LETTER2NAME[letter]}_{loc}.png"
            fig.savefig(outdir / fname, bbox_inches="tight")
            plt.close(fig)


def plot_emotion_h1(b_base: pd.DataFrame, res: pd.DataFrame, outdir: Path):
    base_row = b_base[b_base["inventory"] == "PANASX"]
    if base_row.empty:
        return
    base_vals = {t: base_row[f"t_{t}"].mean() for t in EMOTION_TRAITS}
    for split in [e.lower() for e in EMOTION_TRAITS]:
        sub = res[
            (res["dataset"] == "emotion")
            & (res["use_peft"] == "baseline")
            & (res["phase"] == "post")
            & (res["split"] == split)
        ]
        if sub.empty:
            continue
        post_vals = {t: sub[f"t_{t}"].mean() for t in EMOTION_TRAITS}
        fig, ax = plt.subplots(figsize=(10, 6))
        x = np.arange(len(EMOTION_TRAITS))
        width = 0.35
        ax.bar(x - width / 2, [base_vals[t] for t in EMOTION_TRAITS], width, color="red", label="Pre-trained")
        ax.bar(x + width / 2, [post_vals[t] for t in EMOTION_TRAITS], width, color="green", label="Finetuned")
        ax.set_xticks(x)
        ax.set_xticklabels(EMOTION_TRAITS)
        ax.set_ylabel("Score")
        ax.set_title(f"Emotion scores | Pre-trained vs. Finetuned on {split.capitalize()} split")
        ax.legend()
        fig.tight_layout()
        fname = f"{variant_suffix}_H1_Emotion_{split}.png"
        fig.savefig(outdir / fname, bbox_inches="tight")
        plt.close(fig)


def style_symlog(ax):
    ax.set_xscale("symlog", linthresh=1e-3)
    ax.set_xticks(TICKS)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())


def plot_personality_h2(res: pd.DataFrame, inv: str, outdir: Path):
    for letter in OCEAN_LETTERS:
        trait_lower = LETTER2LOWER[letter]
        for loc in LOCATIONS:
            sub = res[
                (res["dataset"] == "pandora")
                & (res["inventory"] == inv)
                & (res["use_peft"] == "lora")
                & (res["phase"] == "post")
                & (res["split_trait"] == trait_lower)
                & (res["split_location"] == loc)
                & (res["split_size"].isin(SPLIT_SIZES))
            ]
            if sub.empty:
                continue
            fig, ax = plt.subplots(figsize=(8, 6))
            for sz in SPLIT_SIZES:
                d = sub[sub["split_size"] == sz]
                if d.empty:
                    continue
                agg = mean_sd(d, f"t_{letter}", ["lora_scale"])
                xs = agg["lora_scale"]
                ys = agg["mean"]
                ax.plot(xs, ys, marker="o", label=f"{sz}%")
                ax.fill_between(xs, ys - agg["sd"], ys + agg["sd"], alpha=0.15)
            style_symlog(ax)
            ax.set_xlabel("LoRA Scale")
            ax.set_ylabel("Score")
            ax.set_title(f"{LETTER2NAME[letter]} scores | LoRA Scales – {loc.title()} splits ({inv})")
            ax.legend(title="Split size")
            fig.tight_layout()
            fname = f"{variant_suffix}_H2_{inv}_{LETTER2NAME[letter]}_{loc}.png"
            fig.savefig(outdir / fname, bbox_inches="tight")
            plt.close(fig)


def plot_emotion_h2(res: pd.DataFrame, outdir: Path):
    for split in [e.lower() for e in EMOTION_TRAITS]:
        sub = res[
            (res["dataset"] == "emotion")
            & (res["use_peft"] == "lora")
            & (res["phase"] == "post")
            & (res["split"] == split)
        ]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 6))
        colors = plt.cm.tab10.colors
        for i, t in enumerate(EMOTION_TRAITS):
            col = f"t_{t}"
            agg = mean_sd(sub, col, ["lora_scale"])
            xs = agg["lora_scale"]
            ys = agg["mean"]
            ax.plot(xs, ys, marker="o", label=t, color=colors[i % len(colors)])
            ax.fill_between(xs, ys - agg["sd"], ys + agg["sd"], alpha=0.15, color=colors[i % len(colors)])
        style_symlog(ax)
        ax.set_xlabel("LoRA Scale")
        ax.set_ylabel("Score")
        ax.set_title(f"Emotion scores | LoRA Scales – {split.capitalize()} split")
        ax.legend()
        fig.tight_layout()
        fname = f"{variant_suffix}_H2_Emotion_{split}.png"
        fig.savefig(outdir / fname, bbox_inches="tight")
        plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", type=str, default="baseline_pt_scored.csv")
    p.add_argument("--results", type=str, default="outputs_best/merged_results_scored.csv")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    baseline_df = preprocess_baseline(pd.read_csv(args.baseline))
    results_df = preprocess_results(pd.read_csv(args.results))
    plot_personality_h1(baseline_df, results_df, "BFI10", outdir)
    plot_personality_h1(baseline_df, results_df, "IPIP120", outdir)
    plot_emotion_h1(baseline_df, results_df, outdir)
    plot_personality_h2(results_df, "BFI10", outdir)
    plot_personality_h2(results_df, "IPIP120", outdir)
    plot_emotion_h2(results_df, outdir)


if __name__ == "__main__":
    main()
