"""
Module for managing and storing evaluation results in a structured format.
"""
import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Union


class EvalResultsManager:
    """
    Class for handling the storage and retrieval of evaluation results,
    separating them from the main experiment metadata.
    """
    
    @staticmethod
    def save_custom_eval_results(
        output_dir: str,
        experiment_id: str,
        phase: str,
        eval_type: str,
        question: str,
        answers: List[str],
        results: Dict[str, List[Dict[str, Any]]],
        epoch: Optional[int] = None,
        step: Optional[int] = None,
    ) -> str:
        """
        Save custom evaluation results to a CSV file.
        For mid-phase evaluations, it appends to a single consolidated file.
        
        Args:
            output_dir: Base output directory
            experiment_id: Unique experiment ID
            phase: Evaluation phase (pre, post, mid)
            eval_type: Type of evaluation (personality, emotion)
            question: Question used for evaluation
            answers: List of possible answers
            results: Dictionary of results keyed by scale
            epoch: Current epoch (for mid-phase evaluations)
            step: Current step (for mid-phase evaluations)
            
        Returns:
            str: Path to the saved CSV file
        """
        # Create the evals directory if it doesn't exist
        evals_dir = os.path.join(output_dir, experiment_id, "evals")
        os.makedirs(evals_dir, exist_ok=True)
        
        # Prepare filename
        if phase == "mid":
            filename = f"custom_eval_{eval_type}_{phase}.csv"
        else:
            filename = f"custom_eval_{eval_type}_{phase}.csv"
        
        filepath = os.path.join(evals_dir, filename)
        
        # Convert results to DataFrame
        rows = []
        for scale, scale_results in results.items():
            scale_value = scale.replace("scale_", "")
            for result in scale_results:
                row = {
                    "experiment_id": experiment_id,
                    "phase": phase,
                    "eval_type": eval_type,
                    "question": question,
                    "answer": result["answer"],
                    "temperature": result["temp"],
                    "probability": result["prob"],
                    "scale": scale_value,
                }
                
                if epoch is not None:
                    row["epoch"] = epoch
                if step is not None:
                    row["step"] = step
                    
                rows.append(row)
        
        # Create DataFrame
        new_df = pd.DataFrame(rows)
        
        # For mid-phase evaluations, append to existing file if it exists
        if phase == "mid" and os.path.exists(filepath):
            existing_df = pd.read_csv(filepath)
            combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            combined_df.to_csv(filepath, index=False)
        else:
            new_df.to_csv(filepath, index=False)
        
        return filepath
    
    @staticmethod
    def compile_mid_epoch_results(output_dir: str, experiment_id: str, eval_type: str) -> str:
        """
        Compile all mid-epoch evaluation results into a single CSV file.
        This is now a legacy method as we're directly appending to a single file.
        It's kept for compatibility with older code.
        
        Args:
            output_dir: Base output directory
            experiment_id: Unique experiment ID
            eval_type: Type of evaluation (personality, emotion)
            
        Returns:
            str: Path to the compiled CSV file
        """
        evals_dir = os.path.join(output_dir, experiment_id, "evals")
        mid_file = os.path.join(evals_dir, f"custom_eval_{eval_type}_mid.csv")
        
        # If the consolidated file already exists, just return its path
        if os.path.exists(mid_file):
            return mid_file
            
        # For backward compatibility, find and compile any old-style JSON files
        json_files = []
        for filename in os.listdir(evals_dir):
            if filename.startswith("epoch") and "step" in filename and filename.endswith(".json"):
                json_files.append(os.path.join(evals_dir, filename))
        
        if not json_files:
            return None
        
        # Compile results
        all_rows = []
        for json_file in json_files:
            with open(json_file, 'r') as f:
                data = json.load(f)
                
            for result in data.get("results", []):
                row = {
                    "experiment_id": experiment_id,
                    "phase": "mid",
                    "eval_type": eval_type,
                    "epoch": data.get("epoch"),
                    "step": data.get("step"),
                    "scale": data.get("scale"),
                    "temperature": result.get("temp"),
                    "answer": result.get("answer"),
                    "probability": result.get("prob")
                }
                all_rows.append(row)
        
        # Save to CSV
        if all_rows:
            df = pd.DataFrame(all_rows)
            df.to_csv(mid_file, index=False)
            
            # Remove the original JSON files to save space
            for json_file in json_files:
                os.remove(json_file)
            
            return mid_file
        return None
    
    @staticmethod
    def get_evaluation_inventory(dataset_name: str) -> str:
        """
        Determine the appropriate evaluation inventory based on the dataset.
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            str: Name of the inventory to use for evaluation
        """
        if dataset_name.lower() == "pandora":
            return "personality"
        elif dataset_name.lower() == "emotion":
            return "emotion"
        else:
            return "unknown"
