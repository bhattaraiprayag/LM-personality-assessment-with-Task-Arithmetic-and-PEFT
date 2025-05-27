import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import numpy as np
import os

# --- CONFIGURATION ---
DATA_FOLDER = "./outputs_best"  # Change this path
INPUT_CSV = os.path.join(DATA_FOLDER, "merged_results_scored.csv")  # Change this file name
OUTPUT_PNG = os.path.join(DATA_FOLDER, "ocean_heatmap_finetuning_results.png")

# --- Load data ---
df = pd.read_csv(INPUT_CSV)

# --- Trait columns and metadata grouping ---
trait_columns = ['t_O', 't_C', 't_E', 't_A', 't_N']
trait_map = {
    't_O': 'Openness',
    't_C': 'Conscientiousness',
    't_E': 'Extraversion',
    't_A': 'Agreeableness',
    't_N': 'Neuroticism'
}

# --- Aggregate: average across seeds ---
agg_df = df.groupby(['split_trait', 'split_location', 'split_size'])[trait_columns].mean().reset_index()

# --- Create row IDs for heatmap ---
agg_df["row_id"] = agg_df.apply(lambda row: f"{row['split_trait']}-{row['split_location']}-{row['split_size']}", axis=1)
agg_df["trait_order"] = pd.Categorical(agg_df["split_trait"], categories=["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"], ordered=True)
agg_df["location_order"] = pd.Categorical(agg_df["split_location"], categories=["top", "bot"], ordered=True)
agg_df = agg_df.sort_values(["trait_order", "location_order", "split_size"])
row_order = agg_df["row_id"]

# --- Format for heatmap ---
heatmap_data = agg_df.set_index("row_id")[trait_columns].rename(columns=trait_map)
meta_df = agg_df.set_index("row_id").loc[row_order, ["split_trait", "split_location", "split_size"]]
full_heatmap_data = pd.concat([meta_df, heatmap_data], axis=1)

# --- Color mappings (same as original) ---
trait_colors = {
    "openness": "#8dd3c7",
    "conscientiousness": "#ffffb3",
    "extraversion": "#bebada",
    "agreeableness": "#fb8072",
    "neuroticism": "#80b1d3"
}
location_colors = {"top": "#1f78b4", "bot": "#33a02c"}
size_colors = {5: "#a6cee3", 10: "#b2df8a", 15: "#fb9a99"}

# --- Plot heatmap ---
fig, ax = plt.subplots(figsize=(14, 10))
sns.set_theme(style="white")
sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap=sns.diverging_palette(220, 20, as_cmap=True),
    linewidths=0,
    linecolor=None,
    ax=ax,
    yticklabels=False,
    annot_kws={"va": "center", "ha": "center", "fontsize": 7},
)

# --- Add colored rectangles for metadata ---
for i, row_id in enumerate(full_heatmap_data.index):
    for j, (colname, colormap) in enumerate(zip(
        ["split_trait", "split_location", "split_size"],
        [trait_colors, location_colors, size_colors]
    )):
        #You have to identify the issue with bottom, the dataset has bottom, it has to be changed to bot
        if full_heatmap_data.loc[row_id, colname] == "bottom":
            full_heatmap_data.loc[row_id, colname] = "bot"
        color = colormap[full_heatmap_data.loc[row_id, colname]]
        # color = colormap[full_heatmap_data.loc[row_id, colname]]
        rect = patches.Rectangle(
            (j - 3, i), 1, 1,
            linewidth=0,
            facecolor=color,
            transform=ax.transData,
            clip_on=False
        )
        ax.add_patch(rect)

# --- Merged axis labels ---
def add_merged_labels(col, x_pos, rotation=0):
    last_val, start_idx = None, None
    for i, val in enumerate(full_heatmap_data[col]):
        if val != last_val:
            if start_idx is not None:
                mid = (start_idx + i - 1) / 2 + 0.5
                ax.text(x_pos, mid, str(last_val), va='center', ha='center',
                        rotation=rotation, fontsize=10, weight='bold')
            start_idx = i
            last_val = val
    if start_idx is not None:
        mid = (start_idx + len(full_heatmap_data) - 1) / 2 + 0.5
        ax.text(x_pos, mid, str(last_val), va='center', ha='center',
                rotation=rotation, fontsize=10, weight='bold')

add_merged_labels("split_trait", x_pos=-2.5, rotation=45)
add_merged_labels("split_location", x_pos=-1.5)
add_merged_labels("split_size", x_pos=-0.5)

# --- Final tweaks ---
ax.set_xlim(-3, heatmap_data.shape[1])
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
ax.set_title("Mean OCEAN evaluation scores across Pandora splits | Baseline Fine-tuning", fontsize=14)
ax.set_xlabel("")
ax.set_ylabel("")
plt.tight_layout()

# --- Save heatmap ---
plt.savefig(OUTPUT_PNG, dpi=600)
plt.close()
print(f"Saved heatmap to: {OUTPUT_PNG}")
