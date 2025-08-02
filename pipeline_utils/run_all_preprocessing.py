"""
run_all_preprocessing.py

Running all preprocessing steps at once. Run script with:

$python run_all_preprocessing.py --embedding_model <model_name> --n_clusters <int>

Example usage:

$python run_all_preprocessing.py --embedding_model nomic-embed-text --n_clusters 4
"""

import argparse
import os
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from pipeline_utils.download_helper import download_save_hagrid
from pipeline_utils.embedder import Embedder
from pipeline_utils.kmeans_cluster import KMeansCluster
from pipeline_utils.preprocessor import Preprocessor

def parse_args():
    """
    Collecting arguments to do preprocessing.
    """
    parser = argparse.ArgumentParser(
        description="Preprocess data."
    )

    parser.add_argument("--embedding_model", type=str, 
    help="Name of embedding model to be used with Ollama.")

    parser.add_argument("--n_clusters", type=int, 
    help="Number of clusters for clustering.")

    return parser.parse_args()

def run_all_preprocessing():
    args = parse_args()

    if args.n_clusters is None:
        num_clusters = 4
    else:
        num_clusters = args.n_clusters

    # Downloading the hagrid subset from ragbench
    download_save_hagrid()
    df = pd.read_json("data/hagrid_full.jsonl", lines=True)

    # Tokenization and unfrequent word removal with spacy
    preprocessor = Preprocessor(df)

    # Embedding text chunks with nomic-embed-text as default embedding model
    if args.embedding_model is None:
        EMBEDDING_MODEL = "nomic-embed-text"
        initial_embeddings = Embedder(EMBEDDING_MODEL, df)
        additional_embeddings = Embedder("mxbai-embed-large", df)
    else:
        EMBEDDING_MODEL = args.embedding_model
        initial_embeddings = Embedder(EMBEDDING_MODEL, df)

    # Clustering text chunks
    df_embeddings = pd.read_json("data/vector_db.jsonl", lines=True)
    K_range = 50
    self_determine_k = False  # If True, k cluster with highest silhouette score
    verbose = False

    if num_clusters is None:
        num_clusters = 4

    kmeans_cluster = KMeansCluster(df_embeddings, EMBEDDING_MODEL, num_clusters, self_determine_k, K_range, verbose)

if __name__ == "__main__":
    run_all_preprocessing()
