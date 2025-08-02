"""
split_dataset.py

This script splits the dataset - tokenized_db.jsonl & vector_db.jsonl
"""

import os
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

class DatasetSplitter:
    def __init__(self, file_path):
        """
        Initialize the DatasetSplitter with the path to the JSONL file.
        """
        self.file_path = file_path
        self.filename = file_path.split("./data/")[1].removesuffix(".jsonl")
        self.data = pd.read_json(file_path, lines=True)

    def split_dataset(self):
        """
        Split the dataset into training, validation, and test sets based on 'dataset_name'.
        """
        for dataset_name in ["hagrid_train", "hagrid_validation", "hagrid_test"]:
            subset = self.data[self.data["dataset_name"] == dataset_name]
            split_name = dataset_name.split("_")[-1]
            output_file = rf"./data/{self.filename}_{split_name}.jsonl"
            subset.to_json(output_file, orient="records", lines=True)

if __name__ == "__main__":
    # Example usage
    splitter = DatasetSplitter(r"./data/tokenized_db.jsonl")
    splitter.split_dataset()
    splitter = DatasetSplitter(r"./data/vector_db.jsonl")
    splitter.split_dataset()
