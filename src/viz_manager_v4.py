# viz_manager.py

"""
Visualizations manager module for experiment results.

All experiment results are stored in the outputs/ directory, in a json file.

JSON:
[
The JSON structure is organized hierarchically, containing key-value pairs where nested objects provide detailed metrics for different parts of an experiment. Here's a breakdown:

Top-level keys: Each experiment is identified by a unique key (e.g., "ae0011dcdc"), which acts as an identifier for a specific experiment.

First-level properties: Each experiment object contains several properties related to its configuration, such as:

"dataset": The name of the dataset used (e.g., "pandora").
"split": A subset or split of the data (e.g., "agreeableness-bot-5").
"subset": The size of the data subset used (e.g., 1000).
"output", "model_name", "seed", "epochs", "batch_size", "lr" (learning rate), "grad_steps", "use_peft", "scale_peft", "warmup_ratio", "num_workers", "accelerator", "devices", and "optimal_lr": These fields contain various model training configurations and parameters.
Nested objects:

"results": Contains detailed results of the experiment.
Sub-nested objects:
"train_metrics": Stores training-related metrics, including:
"val_loss": Validation loss value.
"val_perplexity": Validation perplexity.
"train_loss": Training loss value.
"test_metrics": Stores testing-related metrics, including:
"test_loss": Testing loss value.
"test_perplexity": Testing perplexity (in this case, a value of "Infinity").
"personality_eval_pre": An array containing personality evaluation before the experiment. Each object within this array includes:
"answer": A personality trait description (e.g., "does a thorough job.").
"value": The trait's corresponding value.
"prob": Probability score of the trait.
"temp": Temperature used during evaluation.
"norm_probs": Normalized probability value.
"personality_eval_post": Similar to "personality_eval_pre", but reflects personality evaluations after the experiment.
Arrays:
"devices": Contains an array of device indices used (e.g., [0]).
The "personality_eval_pre" and "personality_eval_post" are arrays of objects, each representing a different personality evaluation with multiple probability-related attributes.
]

ExperimentTracker module has already generated the necessary CSV files for each experiment, in its respective subdirectory within outputs/.

CSVs (pre/post personality evaluation results):
[
Pre-Fine-Tuning:
                        answer  value         prob  temp  norm_probs
0         does a thorough job.      1 -1093.640906   0.1         0.0
1         gets nervous easily.      1 -1061.910980   0.1         0.0
2   has an active imagination.      1  -947.440833   0.1         0.0
3  has few artistic interests.      1 -1012.702844   0.1         0.0
4       is generally trusting.      1 -1060.935577   0.1         0.0
Post-Fine-Tuning:
                        answer  value         prob  temp  norm_probs
0         does a thorough job.      1 -1093.640906   0.1         0.0
1         gets nervous easily.      1 -1061.910980   0.1         0.0
2   has an active imagination.      1  -947.440833   0.1         0.0
3  has few artistic interests.      1 -1012.702844   0.1         0.0
4       is generally trusting.      1 -1060.935577   0.1         0.0
]

The JSON is intended to load information about each experiment, and then, for each experiment, pre/post personality evaluation results are loaded into DataFrames for visualization. Ensure that the outputs/ directory contains subdirectories for each experiment, each with 'personality_eval_pre.csv' and 'personality_eval_post.csv'.

PROPOSED VISUALIZATIONS:
[
7. Stacked Bar Chart of Normalized Probabilities Across Answers and Temperatures
    Difficulty: Medium
    Objective: Show how the model's total probability distribution over all answers shifts with temperature.
    X-Axis: Temperature values (temp).
    Y-Axis: Cumulative normalized probabilities (stacked up to 1).
    Stacks: Each segment in the bar represents an answer with its norm_probs.
    Data Derivation:
    From JSON:
    Extract temp, answer, and norm_probs from the evaluation data.
    Implementation Steps:
    Data Preparation:
    For each temp, sum the norm_probs across all answers to ensure they sum to 1 (should already be the case).
    Organize data into a format suitable for stacking (e.g., a matrix with temperatures as rows and answers as columns).
    Plotting:
    Use a stacked bar chart with bars representing temperatures.
    Stack segments within each bar according to norm_probs for each answer.
    Visualization Considerations:
    Use a consistent color palette to represent different answers.
    Add labels or a legend to clarify which color corresponds to which answer.
    Purpose and Insights:
    Purpose: Visualize how the model's allocation of probability mass among different personality traits changes with temperature.
    Insights: Observe dominance or suppression of certain traits at different temperatures.

8. Line Chart of Normalized Probabilities vs. Temperature for Each Personality Trait Split
    Difficulty: Medium to Hard
    Objective: Compare how fine-tuning on different personality trait splits affects the model's responses.
    X-Axis: Temperature values (temp).
    Y-Axis: Normalized probabilities (norm_probs).
    Lines: Each line represents a different dataset split (e.g., top 5% agreeableness).
    Data Derivation:
    From JSON:
    Collect data from experiments using different split values.
    For each split, extract norm_probs for a specific answer at various temperatures.
    Implementation Steps:
    Data Aggregation:
    Combine data from multiple experiments, ensuring consistent temp and answer values.
    Create a data frame with columns: temp, norm_probs, split.
    Plotting:
    Plot lines for each split, showing how norm_probs varies with temp.
    Visualization Considerations:
    Use different colors or line styles to distinguish between splits.
    Focus on answers directly related to the trait being split for clearer insights.
    Purpose and Insights:
    Purpose: Understand the impact of fine-tuning on different personality trait extremes.
    Insights: Determine if fine-tuning on a specific trait split accentuates that trait in the model's responses.

9. Matrix of Heatmaps Comparing Different Scale_peft Values
    Difficulty: Hard
    Objective: Provide a comprehensive comparison of how different scale_peft values affect the model's personality evaluations across temperatures and answers.
    Layout: A grid (matrix) where each cell is a heatmap corresponding to a specific scale_peft value.
    Axes within Heatmaps:
    X-Axis: Temperature values (temp).
    Y-Axis: Personality trait answers (answer).
    Color Scale: Normalized probabilities (norm_probs).
    Data Derivation:
    From JSON:
    Extract data from experiments with varying scale_peft values.
    For each scale_peft, prepare a dataset of temp, answer, and norm_probs.
    Implementation Steps:
    Data Organization:
    Create a list of datasets, each corresponding to a different scale_peft value.
    For each dataset, create a pivot table as in Visualization 3.
    Plotting:
    Arrange the heatmaps in a grid, with scale_peft values labeled accordingly.
    Use a consistent color scale across all heatmaps.
    Visualization Considerations:
    Ensure that the axes are labeled clearly.
    Consider interactive visualization tools if the matrix becomes large.
    Purpose and Insights:
    Purpose: Allow simultaneous comparison of the effects of different scale_peft values on the model's personality traits.
    Insights: Identify trends or optimal scale_peft settings that enhance or mitigate certain personality traits.

10. 3D Surface Plot of Normalized Probabilities vs. Temperature and Scale_peft for an Answer
    Difficulty: Hard
    Objective: Visualize the interaction between temp and scale_peft on the model's probability for a specific personality trait answer.
    Axes:
    X-Axis: Temperature values (temp).
    Y-Axis: scale_peft values.
    Z-Axis (Surface Height): Normalized probabilities (norm_probs).
    Data Derivation:
    From JSON:
    Aggregate data from multiple experiments varying scale_peft.
    For a specific answer, collect temp, scale_peft, and norm_probs.
    Implementation Steps:
    Data Preparation:
    Create a grid of temp and scale_peft values.
    Map norm_probs onto this grid.
    Plotting:
    Use a 3D plotting library (e.g., Matplotlib's plot_surface).
    Plot the surface representing norm_probs over the grid.
    Visualization Considerations:
    Rotate the 3D plot to find the best viewing angle.
    Include color mapping to enhance depth perception.
    Purpose and Insights:
    Purpose: Explore the combined effects of temperature and adapter scaling on the model's confidence in a specific personality trait.
    Insights: Understand complex interactions between hyperparameters and model behavior, potentially guiding future fine-tuning strategies.
]

"""

import os
import json
from typing import List, Dict, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mtick
import plotly.graph_objects as go

from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D

OCEAN_ANSWER_KEY = {
    "has an active imagination.": "Openness",
    "has few artistic interests.": "Openness",
    "does a thorough job.": "Conscientiousness",
    "tends to be lazy.": "Conscientiousness",
    "is outgoing, sociable.": "Extraversion",
    "is reserved.": "Introversion",
    "is generally trusting.": "Agreeableness",
    "tends to find fault with others.": "Agreeableness",
    "is relaxed, handles stress well.": "Neuroticism",
    "gets nervous easily.": "Neuroticism",
}

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.
    """
    theta = np.linspace(0, 2 * np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):
        def transform_path_non_affine(self, path):
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):
        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars, radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self, spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5) + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)
    register_projection(RadarAxes)
    return theta


class VizManager:
    def __init__(self, output_dir: str = '../outputs/'):
        self.output_dir = output_dir
        self.metadata_file = os.path.join(self.output_dir, 'valid_experiment_metadata.json')
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """
        Load experiment metadata from the JSON file.
        """
        if not os.path.exists(self.metadata_file):
            raise FileNotFoundError(f"Metadata file not found at {self.metadata_file}")
        with open(self.metadata_file, 'r') as f:
            metadata = json.load(f)
        return metadata

    def _get_experiment_paths(self) -> List[Dict[str, str]]:
        """
        Retrieve paths to pre and post evaluation CSVs for each experiment.
        """
        experiments = []
        for exp_id in self.metadata:
            exp_folder = os.path.join(self.output_dir, exp_id)
            pre_csv = os.path.join(exp_folder, 'personality_eval_pre.csv')
            post_csv = os.path.join(exp_folder, 'personality_eval_post.csv')
            if os.path.exists(pre_csv) and os.path.exists(post_csv):
                experiments.append({
                    'id': exp_id,
                    'pre_csv': pre_csv,
                    'post_csv': post_csv,
                    'metadata': self.metadata[exp_id]
                })
            else:
                print(f"Warning: Missing CSV files for experiment {exp_id}. Skipping.")
        return experiments
    
    def _plot_line_chart(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame):
        plt.figure(figsize=(12, 6))
        for answer in df_pre['answer'].unique():
            df_pre_answer = df_pre[df_pre['answer'] == answer]
            df_post_answer = df_post[df_post['answer'] == answer]

            plt.plot(df_pre_answer['temp'], df_pre_answer['norm_probs'], label=f'Pre - {answer}', linestyle='solid')
            plt.plot(df_post_answer['temp'], df_post_answer['norm_probs'], label=f'Post - {answer}', linestyle='dashed')
        
        split = self.metadata[exp_id]['split']
        use_peft = self.metadata[exp_id]['use_peft'] if 'use_peft' in self.metadata[exp_id] else 'no_peft'
        scale_peft = self.metadata[exp_id]['scale_peft'] if 'scale_peft' in self.metadata[exp_id] else 'no_peft'
        prefix = f'{split}_{use_peft}_{scale_peft}'

        plt.xlabel('Temperature')
        plt.ylabel('Normalized Probabilities')
        plt.title(f'Norm. Probs vs. Temperature (Pre and Post Fine-Tuning) \n{prefix}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_line_chart_temperature_vs_probabilities.png'))
        plt.close()
    
    def _plot_bar_chart(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame, fixed_temp: float = 0.7):
        df_pre_fixed = df_pre[df_pre['temp'] == fixed_temp]
        df_post_fixed = df_post[df_post['temp'] == fixed_temp]

        if df_pre_fixed.empty or df_post_fixed.empty:
            print(f"Warning: No data found for fixed temperature {fixed_temp}. Skipping visualization.")
            return

        plt.figure(figsize=(12, 6))
        width = 0.35
        x = range(len(df_pre_fixed['answer']))

        plt.bar(x, df_pre_fixed['norm_probs'], width, label='Pre-Fine-Tuning')
        plt.bar([i + width for i in x], df_post_fixed['norm_probs'], width, label='Post-Fine-Tuning')

        split = self.metadata[exp_id]['split']
        use_peft = self.metadata[exp_id]['use_peft'] if 'use_peft' in self.metadata[exp_id] else 'no_peft'
        scale_peft = self.metadata[exp_id]['scale_peft'] if 'scale_peft' in self.metadata[exp_id] else 'no_peft'
        prefix = f'{split}_{use_peft}_{scale_peft}'

        plt.xlabel('Personality Trait Answers')
        plt.ylabel('Normalized Probabilities')
        plt.title(f'Norm. Probs for Each Answer at Temp {fixed_temp} (Pre and Post Fine-Tuning) \n{prefix}')
        plt.xticks([i + width/2 for i in x], df_pre_fixed['answer'], rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_bar_chart_temp_{fixed_temp}.png'))
        plt.close()
    
    def _plot_heatmap(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame):
        pre_pivot = df_pre.pivot(index='answer', columns='temp', values='norm_probs')
        post_pivot = df_post.pivot(index='answer', columns='temp', values='norm_probs')

        split = self.metadata[exp_id]['split']
        use_peft = self.metadata[exp_id]['use_peft'] if 'use_peft' in self.metadata[exp_id] else 'no_peft'
        scale_peft = self.metadata[exp_id]['scale_peft'] if 'scale_peft' in self.metadata[exp_id] else 'no_peft'
        prefix = f'{split}_{use_peft}_{scale_peft}'

        fig, axes = plt.subplots(1, 2, figsize=(24, 10), sharey=True)
        for phase, pivot, ax in zip(['Pre', 'Post'], [pre_pivot, post_pivot], axes):
            sns.heatmap(pivot, ax=ax, cmap='viridis', annot=True, fmt=".2f")
            ax.set_title(f'{phase} Fine-Tuning')
            ax.set_xlabel('Temperature')
            ax.set_ylabel('OCEAN Trait Answer')
        plt.suptitle(f'Heatmap of Normalized Probabilities Across Temperatures and Answers\n{prefix}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_heatmap_pre_post.png'))
        plt.close()
    
    def _plot_norm_probab_delta(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame):
        merged_df = pd.merge(df_pre, df_post, on=['temp', 'answer'], suffixes=('_pre', '_post'))
        merged_df['delta_norm_probs'] = merged_df['norm_probs_post'] - merged_df['norm_probs_pre']

        split = self.metadata[exp_id]['split']
        use_peft = self.metadata[exp_id]['use_peft'] if 'use_peft' in self.metadata[exp_id] else 'no_peft'
        scale_peft = self.metadata[exp_id]['scale_peft'] if 'scale_peft' in self.metadata[exp_id] else 'no_peft'
        prefix = f'{split}_{use_peft}_{scale_peft}'

        plt.figure(figsize=(12, 6))
        for answer in merged_df['answer'].unique():
            df_answer = merged_df[merged_df['answer'] == answer]
            plt.plot(df_answer['temp'], df_answer['delta_norm_probs'], label=answer)

        plt.axhline(y=0, color='black', linestyle='--')
        plt.xlabel('Temperature')
        plt.ylabel('Delta Normalized Probabilities')
        plt.title(f'Delta Norm. Probs vs. Temperature (Post - Pre Fine-Tuning)\n{prefix}')
        plt.legend(loc='best')
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_delta_norm_probs_temperature.png'))
        plt.close()

    def _plot_scatter_peft_norm_probs(self, experiments: List[Dict[str, str]], fixed_temp: float = 0.7):
        peft_experiments = [exp for exp in experiments if exp['metadata']['use_peft']]
        if not peft_experiments:
            print("No PEFT experiments found. Skipping scatter plot visualization.")
            return
        peft_norm_prob_answer_collect = {}
        for exp in peft_experiments:
            scale_peft = exp['metadata']['scale_peft']
            pre_df = pd.read_csv(exp['pre_csv'])
            post_df = pd.read_csv(exp['post_csv'])
            pre_df_fixed = pre_df[pre_df['temp'] == fixed_temp]
            post_df_fixed = post_df[post_df['temp'] == fixed_temp]
            for df in [pre_df_fixed, post_df_fixed]:
                for _, row in df.iterrows():
                    answer = row['answer']
                    norm_probs = row['norm_probs']
                    if answer not in peft_norm_prob_answer_collect:
                        peft_norm_prob_answer_collect[answer] = []
                    peft_norm_prob_answer_collect[answer].append((scale_peft, norm_probs))
        prefix = exp['metadata']['split']

        fig, axes = plt.subplots(2, 5, figsize=(24, 12))
        for i, (answer, data) in enumerate(peft_norm_prob_answer_collect.items()):
            row, col = divmod(i, 5)
            ax = axes[row, col]
            scale_peft, norm_probs = zip(*data)
            ax.scatter(scale_peft, norm_probs, label=answer)
            ax.set_title(f'{answer}')
            ax.set_xlabel('Scale PEFT')
            ax.set_ylabel('Normalized Probabilities')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        plt.suptitle(f'Scatter Plot of Normalized Probabilities vs. Scale PEFT at Temp {fixed_temp}\n{prefix}', fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.output_dir, f'{prefix}_scatter_peft_norm_probs_temp_{fixed_temp}.png'))
        plt.close()

    # # def plot_radar_chart(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame, fixed_temp: float = 0.7):
    # def _plot_radar_chart(self, experiments: List[Dict[str, str]], fixed_temp: float = 0.7):
    #     """
    #     6. Stacked Radar Chart of Normalized Probabilities for All Answers at a Fixed Temperature (Pre and Post Fine-Tuning)
    #         Difficulty: Medium
    #         Objective: Provide a holistic view of the model's personality profile before and after fine-tuning.
    #         Axes: Each axis represents an answer (personality trait).
    #         Values: Normalized probabilities (norm_probs).
    #         Data Derivation:
    #         From JSON:
    #         For a chosen temp, extract norm_probs for each answer from both pre and post fine-tuning data.
    #         Implementation Steps:
    #         Data Preparation:
    #         Organize the norm_probs for pre and post fine-tuning into lists corresponding to the axes (answers).
    #         Plotting:
    #         Use a radar chart (also known as a spider chart) to plot the norm_probs.
    #         Plot two lines on the chart, one for pre and one for post fine-tuning.
    #         Visualization Considerations:
    #         Normalize the data so that all values are within the same range (e.g., 0 to 1).
    #         Include labels for each axis to indicate the corresponding personality trait.
    #         Purpose and Insights:
    #         Purpose: Visually compare the overall personality traits of the model before and after fine-tuning in a single, intuitive plot.
    #         Insights: Easily identify which traits have increased or decreased due to fine-tuning.
    #     """
    #     # We generate radar charts for each temperature. So, we will be iterating over temps.
    #     # For each temp, the main plot will have subplots, comparing the radar charts for pre and post fine-tuning for a set of temperatures
    #     # Each answer will be a label/vertex on the radar chart, and the normalized probabilities will be the values
    #     # The major plot will be a big one, and each subplot will be for each temperature
    #     # Each subplot will contain 4 circular lines/value sets, one for pre and one for post fine-tuning (base model), and one for pre and post fine-tuning when use_peft is True
    #     # The radar chart will be normalized to the same range for all subplots
    #     # We will have 1 radar chart for each temperature; there will be 4 lines on each radar chart (pre, post, pre-peft, post-peft). They will also be normalized to the same range, and included in legend. And we will have a common legend for all subplots
    #     # Prepare the data for the radar chart
    #     # For each temp, we will have 4 sets of data: pre, post, pre-peft, post-peft
    #     # We will have 4 radar charts for each temp, one for each set of data
    #     # We will have a common legend for all radar charts
    #     for exp in experiments:
    #         exp_id = exp['id']
    #         pre_df = pd.read_csv(exp['pre_csv'])
    #         post_df = pd.read_csv(exp['post_csv'])

    #         # Filter data for the fixed temperature
    #         pre_fixed = pre_df[pre_df['temp'] == fixed_temp]
    #         post_fixed = post_df[post_df['temp'] == fixed_temp]

    #         if pre_fixed.empty or post_fixed.empty:
    #             print(f"Warning: No data found for temp {fixed_temp} in experiment {exp_id}. Skipping radar chart.")
    #             continue

    #         # Sort the answers to ensure consistent ordering
    #         pre_fixed = pre_fixed.sort_values('answer')
    #         post_fixed = post_fixed.sort_values('answer')

    #         # Extract the necessary data
    #         answers = pre_fixed['answer'].tolist()
    #         norm_probs_pre = pre_fixed['norm_probs'].tolist()
    #         norm_probs_post = post_fixed['norm_probs'].tolist()

    #         # Close the loop for radar chart by appending the first value at the end
    #         answers += [answers[0]]
    #         norm_probs_pre += [norm_probs_pre[0]]
    #         norm_probs_post += [norm_probs_post[0]]

    #         # Create the radar chart
    #         fig = go.Figure()

    #         fig.add_trace(go.Scatterpolar(
    #             r=norm_probs_pre,
    #             theta=answers,
    #             fill='toself',
    #             name='Pre-Fine-Tuning'
    #         ))

    #         fig.add_trace(go.Scatterpolar(
    #             r=norm_probs_post,
    #             theta=answers,
    #             fill='toself',
    #             name='Post-Fine-Tuning'
    #         ))

    #         fig.update_layout(
    #             title=f'Normalized Probabilities at Temp {fixed_temp} for Experiment {exp_id}',
    #             polar=dict(
    #                 radialaxis=dict(
    #                     visible=True,
    #                     range=[0, max(max(norm_probs_pre), max(norm_probs_post)) * 1.1]
    #                 )
    #             ),
    #             showlegend=True
    #         )

    #         # # Save the figure as an HTML file for interactivity
    #         # radar_chart_path = os.path.join(self.output_dir, f'{exp_id}', 'radar_chart.html')
    #         # fig.write_html(radar_chart_path)

    #         # Optionally, save as a static image (requires Kaleido)
    #         # Uncomment the following lines if you have Kaleido installed
    #         fig.write_image(os.path.join(self.output_dir, f'{exp_id}', 'radar_chart.png'))

    #         print(f"Radar chart saved for experiment {exp_id} at temperature {fixed_temp}.")

    def _plot_radar_chart(self, exp_id: str, df_pre: pd.DataFrame, df_post: pd.DataFrame, temps: List[float]):
        """
        Plot a radar chart comparing pre and post fine-tuning normalized probabilities at a fixed temperature.
        
        Parameters:
        - exp_id: Identifier for the experiment.
        - df_pre: DataFrame containing pre fine-tuning data.
        - df_post: DataFrame containing post fine-tuning data.
        - fixed_temp: The temperature value at which to plot the radar chart.
        """
        split = self.metadata[exp_id]['split']
        use_peft = self.metadata[exp_id]['use_peft'] if 'use_peft' in self.metadata[exp_id] else 'no_peft'
        scale_peft = self.metadata[exp_id]['scale_peft'] if 'scale_peft' in self.metadata[exp_id] else 'no_peft'
        prefix = f'{split}_{use_peft}_{scale_peft}'

        for fixed_temp in temps:
            # Filter data for the fixed temperature
            df_pre_fixed = df_pre[df_pre['temp'] == fixed_temp]
            df_post_fixed = df_post[df_post['temp'] == fixed_temp]

            if df_pre_fixed.empty or df_post_fixed.empty:
                print(f"Warning: No data found for fixed temperature {fixed_temp} in experiment {exp_id}. Skipping radar chart.")
                return

            # Ensure both DataFrames have the same answers and order
            df_pre_fixed = df_pre_fixed.sort_values('answer')
            df_post_fixed = df_post_fixed.sort_values('answer')

            answers = df_pre_fixed['answer'].tolist()
            pre_probs = df_pre_fixed['norm_probs'].tolist()
            post_probs = df_post_fixed['norm_probs'].tolist()

            # Number of variables
            N = len(answers)

            # Create radar chart
            theta = radar_factory(N, frame='circle')
            spoke_labels = answers

            # Close the plot by appending the first value at the end
            pre_probs += pre_probs[:1]
            post_probs += post_probs[:1]
            theta = np.concatenate((theta, [theta[0]]))

            fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='radar'))
            colors = ['b', 'r']
            labels = ['Pre Fine-Tuning', 'Post Fine-Tuning']
            alphas = [0.25, 0.25]

            ax.plot(theta, pre_probs, color=colors[0], linewidth=2, label=labels[0])
            ax.fill(theta, pre_probs, facecolor=colors[0], alpha=alphas[0])

            ax.plot(theta, post_probs, color=colors[1], linewidth=2, label=labels[1])
            ax.fill(theta, post_probs, facecolor=colors[1], alpha=alphas[1])

            ax.set_varlabels(spoke_labels)

            # Add legend and title
            plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
            plt.title(f'Radar Chart - Norm. Probs at Temperature {fixed_temp}\{prefix}', size=14, position=(0.5, 1.1), ha='center')

            # Save the figure
            radar_chart_path = os.path.join(self.output_dir, f'{prefix}_radar_chart_temp_{fixed_temp}.png')
            plt.savefig(radar_chart_path, bbox_inches='tight')
            plt.close()
            # print(f"Radar chart saved to {radar_chart_path}")

    def generate_visualizations(self):
        """
        Generate visualizations for each experiment.
        """
        temps = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        experiments = self._get_experiment_paths()
        for temp in temps:
            self._plot_scatter_peft_norm_probs(experiments, fixed_temp=temp)
        # print(f"Experiments: {experiments}")
        for exp in experiments:
            print(f"Generating visualizations for Experiment ID: {exp['id']}")
            pre_df = pd.read_csv(exp['pre_csv'])
            post_df = pd.read_csv(exp['post_csv'])
        #     # print(f"Pre-Fine-Tuning Data:\n{pre_df.head()}")
        #     # print(f"Post-Fine-Tuning Data:\n{post_df.head()}")
            self._plot_line_chart(exp['id'], pre_df, post_df)
            for temp in temps:
                self._plot_bar_chart(exp['id'], pre_df, post_df, fixed_temp=temp)
            self._plot_heatmap(exp['id'], pre_df, post_df)
            self._plot_norm_probab_delta(exp['id'], pre_df, post_df)
            # # for temp in temps:
            # #     self._plot_scatter_scale_peft(exp['id'], pre_df, post_df, fixed_temp=temp)
            self._plot_radar_chart(exp['id'], pre_df, post_df, temps)

        # # Check for peft experiments
        # peft_experiments = [exp for exp in experiments if exp['metadata']['use_peft']]
        # # print(f"PEFT Experiments: {peft_experiments} | Total: {len(peft_experiments)}")
        # # print(f"Total {len(experiments)} experiments processed.")


if __name__ == '__main__':
    viz_manager = VizManager()
    viz_manager.generate_visualizations()
    print("All visualizations generated successfully.")
