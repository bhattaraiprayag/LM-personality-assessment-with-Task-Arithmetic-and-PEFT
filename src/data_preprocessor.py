# data_preprocessor.py

"""
Data preprocessing module for personality assessment.
"""

import os
import re
import time
from typing import List, Optional

import ftfy
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataPreprocessor:
    """
    A class to preprocess data for personality assessment.
    """

    CHARACTER_REPLACEMENTS = {
        "â€œ": "“",  # Left double quotation mark
        "â€": "”",  # Right double quotation mark
        "â€˜": "‘",  # Left single quotation mark
        "â€™": "’",  # Right single quotation mark
        "â€“": "–",  # En dash
        "â€”": "—",  # Em dash
        "â€¢": "•",  # Bullet point
        "âˆ™": "•",  # Bullet point
        'â"¢': "™",  # Trademark symbol
        "â„¢": "™",  # Trademark symbol
        "Ã©": "é",  # Latin small letter e with acute
        "â€¦": "…",  # Ellipsis
        "â€": "”",  # Incorrectly encoded double quote
        "Â": "",  # Non-breaking space or similar placeholder
        "Ã": "í",  # Common misencoding for accented letters
        "Ã¼": "ü",  # Latin small letter u with umlaut
    }

    def __init__(
        self,
        data_folder: str,
        comments_file: str,
        author_profiles_file: str,
        splits_folder_balanced: Optional[str] = None,
        ocean_traits: Optional[List[str]] = None,
        mbti_traits: Optional[List[str]] = None,
        k_values: Optional[List[int]] = None,
    ):
        self.data_folder = data_folder
        self.comments_file = comments_file
        self.author_profiles_file = author_profiles_file
        self.splits_folder_balanced = splits_folder_balanced or os.path.join(
            data_folder, "splits_balanced/"
        )
        self.ocean_traits = ocean_traits or [
            "agreeableness",
            "openness",
            "conscientiousness",
            "extraversion",
            "neuroticism",
        ]
        self.mbti_traits = mbti_traits or [
            "introverted",
            "intuitive",
            "thinking",
            "perceiving",
        ]
        self.k_values = k_values or list(range(5, 51, 5))
        self.comments = None
        self.author_profiles = None
        self.filtered_comments = None
        self.ocean_comments = None
        self.mbti_comments = None
        self.all_traits_comments = None

    def load_data(self):
        """
        Load comments and author profiles from CSV files.
        """
        comments_path = os.path.join(self.data_folder, self.comments_file)
        author_profiles_path = os.path.join(self.data_folder, self.author_profiles_file)
        print(f"Loading comments from {comments_path}")
        self.comments = pd.read_csv(comments_path)
        print(f"Loading author profiles from {author_profiles_path}")
        self.author_profiles = pd.read_csv(author_profiles_path)

    def clean_comments(self):
        """
        Clean the comments by removing unwanted characters and fixing encoding issues.
        """
        print("Cleaning comments...")
        self.comments = self.comments.dropna(subset=["body"])
        self.comments = self.comments[self.comments["body"] != ""]
        tqdm.pandas()
        self.comments["body"] = self.comments["body"].progress_apply(self.clean_comment)

    @staticmethod
    def clean_comment(comment):
        """
        Clean a single comment by removing HTML tags, URLs, and fixing character encodings.
        """
        comment = re.sub(r"<[^>]+>", "", comment)
        comment = re.sub(r"http[s]?://\S+", "", comment)
        for old_char, new_char in DataPreprocessor.CHARACTER_REPLACEMENTS.items():
            comment = comment.replace(old_char, new_char)
        comment = ftfy.fix_text(comment)
        comment = re.sub(r"\s+", " ", comment.strip())
        return comment

    def filter_authors_with_profiles(self):
        """
        Filter comments to include only those from authors with available profiles.
        """
        print("Filtering authors with profiles...")
        authors_in_comments = set(self.comments["author"].unique())
        authors_in_profiles = set(self.author_profiles["author"].unique())
        common_authors = authors_in_comments.intersection(authors_in_profiles)
        self.filtered_comments = self.comments[
            self.comments["author"].isin(common_authors)
        ]
        print(f"Total unique authors in comments: {len(authors_in_comments)}")
        print(f"Total unique authors in profiles: {len(authors_in_profiles)}")
        print(f"Number of common authors: {len(common_authors)}")
        rows_removed = len(self.comments) - len(self.filtered_comments)
        print(f"Number of rows removed from comments: {rows_removed}")

    def filter_authors_with_complete_traits(self):
        """
        Further filter authors to include only those with complete OCEAN and MBTI traits.
        """
        print("Filtering authors with complete OCEAN and MBTI traits...")
        ocean_complete_authors = set(
            self.author_profiles.dropna(subset=self.ocean_traits)["author"].unique()
        )
        mbti_complete_authors = set(
            self.author_profiles.dropna(subset=self.mbti_traits)["author"].unique()
        )
        self.ocean_comments = self.filtered_comments[
            self.filtered_comments["author"].isin(ocean_complete_authors)
        ]
        self.mbti_comments = self.filtered_comments[
            self.filtered_comments["author"].isin(mbti_complete_authors)
        ]
        all_traits = self.ocean_traits + self.mbti_traits
        all_complete_authors = set(
            self.author_profiles.dropna(subset=all_traits)["author"].unique()
        )
        self.all_traits_comments = self.filtered_comments[
            self.filtered_comments["author"].isin(all_complete_authors)
        ]
        print(f"Number of authors with all OCEAN traits: {len(ocean_complete_authors)}")
        print(f"Number of authors with all MBTI traits: {len(mbti_complete_authors)}")
        print(f"Number of authors with all traits: {len(all_complete_authors)}")
        print(
            f"Total comments from authors with all OCEAN traits: {len(self.ocean_comments)}"
        )
        print(
            f"Total comments from authors with all MBTI traits: {len(self.mbti_comments)}"
        )
        print(
            f"Total comments from authors with all traits: {len(self.all_traits_comments)}"
        )

    def get_top_bottom_percentile_comments(self, trait, percentile, comments, top=True):
        """
        Retrieve top or bottom percentile of comments based on a specific trait.
        """
        sorted_authors = self.author_profiles[
            ~self.author_profiles[trait].isnull()
        ].sort_values(by=trait, ascending=not top)
        total_authors = len(sorted_authors)
        num_authors = int(np.ceil((percentile / 100) * total_authors))
        percentile_authors = (
            sorted_authors["author"].head(num_authors)
            if top
            else sorted_authors["author"].tail(num_authors)
        )
        filtered_comments = comments[comments["author"].isin(percentile_authors)]
        return filtered_comments

    def perform_splitting(self):
        """
        Split the data into balanced top and bottom percentile datasets and save them.
        """
        print("Performing splitting and saving balanced datasets...")
        if not os.path.exists(self.splits_folder_balanced):
            os.makedirs(self.splits_folder_balanced)
        for k in self.k_values:
            print("=" * 50)
            print(f"Processing K: {k}%")
            top_k_comments_dict = {}
            bottom_k_comments_dict = {}
            for trait in self.ocean_traits:
                top_k_comments = self.get_top_bottom_percentile_comments(
                    trait, k, self.ocean_comments, top=True
                )
                bottom_k_comments = self.get_top_bottom_percentile_comments(
                    trait, k, self.ocean_comments, top=False
                )
                top_k_comments_dict[trait] = top_k_comments
                bottom_k_comments_dict[trait] = bottom_k_comments
            min_top_comments_count = min(
                len(top_k_comments_dict[trait]) for trait in self.ocean_traits
            )
            min_bottom_comments_count = min(
                len(bottom_k_comments_dict[trait]) for trait in self.ocean_traits
            )
            min_comment_count = min(min_top_comments_count, min_bottom_comments_count)
            print(f"Minimum comments count for K={k}%: {min_comment_count}")
            for trait in self.ocean_traits:
                balanced_top_k_comments = top_k_comments_dict[trait].sample(
                    n=min_comment_count, random_state=42
                )
                balanced_bottom_k_comments = bottom_k_comments_dict[trait].sample(
                    n=min_comment_count, random_state=42
                )
                current_split_folder = self.splits_folder_balanced
                if not os.path.exists(current_split_folder):
                    os.makedirs(current_split_folder)
                top_file_name = f"{trait}-top-{k}.csv"
                bottom_file_name = f"{trait}-bot-{k}.csv"
                top_file_path = os.path.join(current_split_folder, top_file_name)
                balanced_top_k_comments.to_csv(top_file_path, index=False, encoding="utf-8")
                bottom_file_path = os.path.join(current_split_folder, bottom_file_name)
                balanced_bottom_k_comments.to_csv(bottom_file_path, index=False, encoding="utf-8")
                print(
                    f"Saved balanced top {k}% comments for {trait} to {top_file_path}"
                )
                print(
                    f"Saved balanced bottom {k}% comments for {trait} to {bottom_file_path}"
                )
            time.sleep(1)

    def save_comments(self):
        """
        Save the filtered comments to CSV files.
        """
        print("Saving filtered comments...")
        ocean_comments_path = os.path.join(
            self.data_folder, "filtered_ocean_comments.csv"
        )
        mbti_comments_path = os.path.join(
            self.data_folder, "filtered_mbti_comments.csv"
        )
        all_traits_comments_path = os.path.join(
            self.data_folder, "filtered_all_traits_comments.csv"
        )
        self.ocean_comments.to_csv(ocean_comments_path, index=False, encoding="utf-8")
        self.mbti_comments.to_csv(mbti_comments_path, index=False, encoding="utf-8")
        self.all_traits_comments.to_csv(all_traits_comments_path, index=False, encoding="utf-8")
        print(f"Saved OCEAN comments to {ocean_comments_path}")
        print(f"Saved MBTI comments to {mbti_comments_path}")
        print(f"Saved comments with all traits to {all_traits_comments_path}")

    def run(self):
        """
        Execute the full data preprocessing pipeline.
        """
        self.load_data()
        self.clean_comments()
        self.filter_authors_with_profiles()
        self.filter_authors_with_complete_traits()
        self.save_comments()
        self.perform_splitting()


if __name__ == "__main__":
    WORKSPACE_PATH = "/pfs/work7/workspace/scratch/ma_pbhattar-kdd_cup_2023/"
    CURRENT_FOLDER = (
        WORKSPACE_PATH
        + "thesis/LM personality assessment with Task Arithmetic and PEFT/"
    )
    DATA_FOLDER = os.path.join(CURRENT_FOLDER, "data/pandora/")
    COMMENTS_FILE = "all_comments_since_2015.csv"
    AUTHOR_PROFILES_FILE = "author_profiles.csv"
    preprocessor = DataPreprocessor(DATA_FOLDER, COMMENTS_FILE, AUTHOR_PROFILES_FILE)
    preprocessor.run()
