import os
import json
import pandas as pd

OCEAN_TRAIT_ANSWER_KEYS = {
    'O (+)': 'has an active imagination.',
    'O (-)': 'has few artistic interests.',
    'C (+)': 'does a thorough job.',
    'C (-)': 'tends to be lazy.',
    'E (+)': 'is outgoing, sociable.',
    'E (-)': 'is reserved.',
    'A (+)': 'is generally trusting.',
    'A (-)': 'tends to find fault with others.',
    'N (+)': 'is relaxed, handles stress well.',
    'N (-)': 'gets nervous easily.',
}

OCEAN_TRAITS = {
    'O': 'Openness',
    'C': 'Conscientiousness',
    'E': 'Extraversion',
    'A': 'Agreeableness',
    'N': 'Neuroticism',
}

class ExperimentTracker:
    def __init__(self, metadata_file, output_dir, excel_file='experiment_results.xlsx'):
        self.metadata_file = metadata_file
        self.output_dir = output_dir
        self.excel_file = os.path.join(self.output_dir, excel_file)
        self.metadata = self._load_metadata()

    def _load_metadata(self):
        with open(self.metadata_file, 'r') as file:
            return json.load(file)

    def _extract_args(self, experiment_data):
        """Extract arguments from the experiment data for the Excel sheet."""
        args = experiment_data.copy()
        results = args.pop('results', None)  # Remove results from arguments for Excel
        return args, results

    def update_excel(self):
        """Update the Excel file with experiment arguments."""
        records = []
        for exp_id, exp_data in self.metadata.items():
            args, _ = self._extract_args(exp_data)
            command = self._generate_command(args)
            args['id'] = exp_id
            args['command'] = command
            records.append(args)

        df = pd.DataFrame(records)
        # Write to Excel
        if os.path.exists(self.excel_file):
            existing_df = pd.read_excel(self.excel_file)
            df = pd.concat([existing_df, df]).drop_duplicates(subset='id', keep='last')

        df.to_excel(self.excel_file, index=False)
        print(f"Updated {self.excel_file} with experiment data.")

    def _generate_command(self, args):
        """Generate the Python command that corresponds to the experiment."""
        command = f"python start_experiment.py --dataset {args['dataset']} --split {args['split']} "
        command += f"--output {args['output']} --model_name {args['model_name']} --seed {args['seed']} "
        command += f"--epochs {args['epochs']} --batch_size {args['batch_size']} --lr {args['lr']} "
        if args.get('use_peft'):
            command += f"--use_peft {args['use_peft']} "
        if args.get('scale_peft'):
            command += f"--scale_peft {args['scale_peft']} "
        return command

    def save_evaluation_results(self):
        """Save pre- and post-finetuning personality evaluation results as separate files."""
        for exp_id, exp_data in self.metadata.items():
            _, results = self._extract_args(exp_data)
            if results:
                self._save_pre_post_results(exp_id, results['personality_eval_pre'], 'pre')
                self._save_pre_post_results(exp_id, results['personality_eval_post'], 'post')

    def _save_pre_post_results(self, exp_id, eval_data, phase):
        """Save the evaluation results for pre- or post-finetuning."""
        output_path = os.path.join(self.output_dir, exp_id, f"personality_eval_{phase}.csv")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df = pd.DataFrame(eval_data)
        # Now, we populate two new columns, based on OCEAN_TRAIT_ANSWER_KEYS and OCEAN_TRAITS:
        # - 'trait + polarity' will be the key in OCEAN_TRAIT_ANSWER_KEYS
        # - net_probs: for each train in OCEAN_TRAITS, we calculate the net probability (for eg.: O (+) - O (-))
        for trait, trait_name in OCEAN_TRAITS.items():
            for polarity in ['(+)', '(-)']:
                key = f"{trait} {polarity}"
                df[key] = df.apply(
                    lambda row: row[OCEAN_TRAIT_ANSWER_KEYS[key]] - row[OCEAN_TRAIT_ANSWER_KEYS[f"{trait} {polarity}"]],
                    axis=1
                )
        df.to_csv(output_path, index=False)
        print(f"Saved {phase}-finetuning evaluation results to {output_path}.")

    def run(self):
        """Run the experiment tracker to update Excel and save results."""
        self.update_excel()
        self.save_evaluation_results()

if __name__ == '__main__':
    # Path to your JSON file and output directory
    output_dir = '../outputs/'
    metadata_file = f"{output_dir}valid_experiment_metadata.json"

    # Initialize and run the experiment tracker
    tracker = ExperimentTracker(metadata_file, output_dir)
    tracker.run()
