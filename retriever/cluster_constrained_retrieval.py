"""
cluster_constrained_retrieval.py

Retriever based on cosine similarity between embeddings. To reduce the search space a classifier is used that assigns most probable top k clusters to a given input query.
"""

import pandas as pd
from typing import List
import torch
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from retriever.cosine_embeddings import CosineEmbeddings
from pipeline_utils.bert_trainer import BERTTrainer

class ClusterConstrainedRetrieval(CosineEmbeddings):
    """
    Inherits from CosineEmbeddings with the constraint that the data used to calculate the cosine similarity between query embedding and document embeddings will be selected based on the provided cluster labels.
    """
    def __init__(self, EMBEDDING_MODEL: str, vector_db: pd.DataFrame):
        self.embedding_model = EMBEDDING_MODEL
        self.vector_db = vector_db
    
    def choose_from_cluster(self, clusters: List[int]):
        """
        Reduces data to those rows that match the provided cluster labels.

        Args:
            df, dataframe such as vector_db.jsonl from which rows can be        
                selected
            clusters, this should contain relevant cluster labels assigned by 
                the classifier such as [3, 0, 1]
        """
        self.vector_db = self.vector_db[self.vector_db["cluster"].isin(clusters)]



class BERTClassifier(BERTTrainer):
    """
    Inherist from BERTTrainer and is used to classify a single query.
    """

    def __init__(self, data: pd.DataFrame, model_path: str):
        super().__init__(data)
        self.best_model_path = model_path

    def assign_cluster(self, query: str, top_k) -> List[int]:
        """
        Given an input query the classifier will predict a corresponding cluster label.

        Args:
            query, from the input
            top_k, allows to control the top k predictions by the model
        
        Returns:
            list of top_k cluster labels predicted by the classifier
        """
        self.model.load_state_dict(torch.load(self.best_model_path, weights_only=True))  # Load the best model
        self.model.eval()

        # Preparing the query
        tokenizer = self.tokenizer
        encoded = tokenizer(
            query,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # shape: [1, num_classes]
            probs = torch.softmax(logits, 1)

            # List of top-k class indices in descending order
            topk = torch.topk(probs, k=top_k, dim=1)
            topk_indices = topk.indices.squeeze(0).cpu().tolist()  

        return topk_indices


if __name__ == "__main__":
    """
    Example usage of cluster constrained retrieval.
    """
    
    EMBEDDING_MODEL = "nomic-embed-text"
    vector_db = pd.read_json("data/vector_db.jsonl", lines=True)
    test_question = "When was activity theory created?"
    top_k_clusters = 2
    top_k_retrievals = 1

    # Classficication of query
    model_path = "model/best_BERT_model.pth"
    #  BERTClassifier(BERTTrainer) requires a pd dataframe for instantiation
    classifier = BERTClassifier(vector_db, model_path)
    clusters = classifier.assign_cluster(test_question, top_k_clusters)
    print(f"For the question: \n\n'{test_question}'\n\nthe top {top_k_clusters} cluster(s) are {clusters}.")

    # Retrieving top documents within a reduced search space based on probable clusters
    retriever = ClusterConstrainedRetrieval(EMBEDDING_MODEL, vector_db)
    retriever.choose_from_cluster(clusters)  # Minimize search space
    retrieved_docs = retriever.score(test_question, top_k_retrievals)

    # Output of retrieved chunks
    print("\nRetrieved chunks:")
    for i, (chunk_id, chunk_text) in enumerate(zip(retrieved_docs["id"], retrieved_docs["chunk"])):
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(f"{chunk_text[0][:200]}..." )
        print("-" * 90)
