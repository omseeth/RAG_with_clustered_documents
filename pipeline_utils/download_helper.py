"""
download_helper.py

This script fetches the hagrid subset of the ragbench dataset from HugginFace. The dataset's splits are concatenated and saved as to .json file, at data/hagrid_full.json.
"""

import os
import sys

from datasets import load_dataset, concatenate_datasets

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def download_save_hagrid(drop_duplicate: bool = True):
    """
    Downloads the hagrid subset from rungalileo/ragbench with HuggingFace's datasets class. The dataset's splits are concatenated and saved to .jsonl file, at data/hagrid_full.jsonl.

    Args:
        drop_duplicate: if False all data points from hagrid are saved to hagrid_full.jsonl
    """
    ds = load_dataset("rungalileo/ragbench", "hagrid")

    train_set = ds["train"]
    test_set = ds["test"]
    eval_set = ds["validation"]

    full_dataset = concatenate_datasets([train_set, test_set, eval_set])
    df = full_dataset.to_pandas()

    # The hagrid subset contains duplicates that should be dropped
    if drop_duplicate:
        df = df.drop_duplicates(subset="question")

    #  Ensuring the "data" folder exists
    os.makedirs("data", exist_ok=True)
    df.to_json("data/hagrid_full.jsonl", orient="records", lines=True)
    print("Downloaded and saved the hagrid subset of ragbench from rungalileo/ragbench on Hugginface to data/hagrid_full.jsonl.")


if __name__ == "__main__":
    """
    Example of downloading the hagrid dataset.
    """

    download_save_hagrid()
    