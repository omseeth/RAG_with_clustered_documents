"""
retriever_eval.py

This script contains success@k, mean reciprocal rank (MRR), and average latency evaluations for the retrieval part of the RAG pipeline. A separate evaluation for retrievers combined through rank fusion is available at evaluation/rrf_weight.py. Run script with the following options:

$python evaluation/retriever_eval.py --retriever <retriever_name> --embedding_model <model_name> <int> --top_k --top_clusters <int> --max_clusters <int>

Example usage:

# BM15
$python retriever_eval.py --retriever bm25 --k 3

# Cosine similarity for embeddings
$python retriever_eval.py --retriever cosine --embedding_model mxbai-embed-large --k 1

# Cosine similarity for embeddings with clustering constraints based on MoBERT
$python retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 1 --top_clusters 2

# Cosine similarity for embeddings with clustering constraints based on FAISS
$python retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 1 --top_clusters 2 --max_clusters 4
"""

import argparse
import os
import sys
import time
from typing import Tuple 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from tqdm import tqdm
import spacy

from retriever.BM25 import BM25
from retriever.cosine_embeddings import CosineEmbeddings
from retriever.cluster_constrained_retrieval import BERTClassifier, ClusterConstrainedRetrieval
from retriever.cluster_constrained_FAISS import ClusterConstrainedFAISS

def parse_args():
    """
    Collecting arguments to run evaluation of retrieval methods.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate success@k of retrieval."
    )

    parser.add_argument("--retriever", type=str, choices=["bm25", "cosine", "MoBERT", "faiss"], help="Retriever mode to use for evaluation.")

    parser.add_argument("--embedding_model", type=str, 
    help="Name of embedding model to be used with Ollama.")

    parser.add_argument("--k", type=int, help="Number of top k retrieved documents for evaluation.")

    parser.add_argument("--top_clusters", type=int, help="Number of top n cluster to consider for retrieval.")

    parser.add_argument("--max_clusters", type=int, help="Number of maximal clusters, e.g. according to the silhouette coefficient.")

    return parser.parse_args()

def retriever_evaluation(retriever_mode: str, embedding_model: str, top_k: int, top_clusters: int, max_clusters: int) -> Tuple[float, float, float]:
    """
    Success@k measures whether the document is in a list of size k of documents. MRR measures the quality of the ranking of retrieved documents.

    Args:
        str, to indicate the type of retrieval method
        cluster_n, number of cluster to consider for cluster constrained retrieval
        top_k, the number of documents to be retrieved
        embedding_model, name of Ollama embedding model for retrieval

    Returns:
        tuple, with floats of success rate and average latency
    """
    EMBEDDING_MODEL = embedding_model
    nlp = spacy.load("en_core_web_sm")

    tokenized_db = pd.read_json("data/tokenized_db.jsonl", lines=True)
    vector_db = pd.read_json("data/vector_db.jsonl", lines=True)

    if top_clusters is not None and max_clusters is not None:
        if top_clusters > max_clusters:
            print("Top clusters cannot exceed the number of max clusters.")

    # Instantiating different retrievers
    
    #BM25 retriever
    bm25_retriever = BM25(tokenized_db)

    # Embedding based retriever with cosine similarity
    embedding_retriever = CosineEmbeddings(EMBEDDING_MODEL, vector_db)

    # Clustering based retriever with ModernBERT classifier
    classifier_path = "model/best_BERT_model.pth"
    if os.path.exists(classifier_path):
        df_test = vector_db[vector_db["split"]=="test"]
        classifier = BERTClassifier(df_test, classifier_path)
        MoBERT_retriever = ClusterConstrainedRetrieval(EMBEDDING_MODEL, vector_db) # This object needs to be instantiated with vector_db
    else:
        print("Please train a classifier before running the evaluation.")

    # Clustering based retriever with FAISS
    if max_clusters is None:
        max_clusters = 4

    faiss_retriever = ClusterConstrainedFAISS(EMBEDDING_MODEL, vector_db, max_clusters, top_clusters)

    # ModernBERT classifier was trained on the training set, evaluation should be done with a test subset
    if retriever_mode == "MoBERT":
        # For testing clustering constraints
        questions = df_test["question"]
        ground_truth = df_test["id"]
    else:
        questions = vector_db["question"]
        ground_truth = vector_db["id"]

    n_correct = 0
    reciprocal_ranks = []
    n_chunks = 0
    latencies = []

    # Evaluate success, MRR, and latency of the retrieval
    for question, true_chunk in tqdm(zip(questions, ground_truth), desc="Evaluating success@k, MRR, latencies", total=len(questions)):
        
        start = time.time()  # Time before the retrieval

        # BM25 retrieval
        if retriever_mode == "bm25":
            # Preprocessing query with spacy
            tokenized_query = nlp(question)
            filtered_query = [
                token.text for token in tokenized_query
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            retrieved_docs = bm25_retriever.search(filtered_query, top_k)

            end = time.time() # Time at the end of retrieval
            latency = end - start
            latencies.append(latency)
            
            if true_chunk in retrieved_docs["id"]:
                n_correct += 1
                rank = retrieved_docs["id"].index(true_chunk) + 1 
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)

        # Retrieval based on cosine similarity between embeddings
        elif retriever_mode == "cosine":
            retrieved_docs = embedding_retriever.score(question, top_k)

            end = time.time() # Time at the end of retrieval
            latency = end - start
            latencies.append(latency)

            if true_chunk in retrieved_docs["id"]:
                n_correct += 1
                rank = retrieved_docs["id"].index(true_chunk) + 1 
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        
        # Retrieval based on pre-clustering of embeddings with ModernBERT classifier
        elif retriever_mode == "MoBERT":
            # The retriever object is conditioned upon the number of clusters
            clusters = classifier.assign_cluster(question, top_clusters)
            MoBERT_retriever.choose_from_cluster(clusters)
            retrieved_docs = MoBERT_retriever.score(question, top_k)
        
            end = time.time() # Time at the end of retrieval
            latency = end - start
            latencies.append(latency)

            if true_chunk in retrieved_docs["id"]:
                n_correct += 1
                rank = retrieved_docs["id"].index(true_chunk) + 1 
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        
        # Retrieval based on pre-clustering of embeddings with FAISS
        elif retriever_mode == "faiss":
            retrieved_docs = faiss_retriever.score(question, top_k)
        
            end = time.time() # Time at the end of retrieval
            latency = end - start
            latencies.append(latency)
            
            if true_chunk in retrieved_docs["id"]:
                n_correct += 1
                rank = retrieved_docs["id"].index(true_chunk) + 1 
                reciprocal_ranks.append(1 / rank)
            else:
                reciprocal_ranks.append(0)
        
        n_chunks += 1
        
        # Restore to full vector_db for next iteration
        MoBERT_retriever.vector_db = vector_db

    success = n_correct / n_chunks
    mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
    average_latency = sum(latencies) / len(latencies)
    
    return (success, mrr, average_latency)

if __name__ == "__main__":
    args = parse_args()
    retriever_mode = args.retriever
    top_k = args.k
    top_clusters = args.top_clusters
    max_clusters = args.max_clusters
    embedding_model = args.embedding_model

    if embedding_model is None:
        embedding_model = "nomic-embed-text"

    if retriever_mode and top_k is not None:

        success_rate, mrr, avg_latency = retriever_evaluation(retriever_mode, embedding_model, top_k, top_clusters, max_clusters)

        print(f"Success of retrieval with method {retriever_mode} for k={top_k} is {success_rate} with a MRR of {mrr} and an average latency of {avg_latency}.")
