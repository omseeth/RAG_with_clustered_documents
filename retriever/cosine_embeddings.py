"""
cosine_embeddings.py

The CosineEmbeddings class has profited from code taken from https://huggingface.co/blog/ngxson/make-your-own-rag.
"""

import ollama
import os
import pandas as pd
import sys
from typing import List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class CosineEmbeddings:
    """
    Retriever based on cosine similarities between document embeddings.
    """

    def __init__(self, EMBEDDING_MODEL: str, vector_db: pd.DataFrame):
        self.embedding_model = EMBEDDING_MODEL
        self.vector_db = vector_db

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """ 
        Simple method to calculate the cosine similarity between two vectors a and b.

        Args:
            a: embedding vector
            b: embedding vector
        
        Returns:
            Cosine similarity
        """
        dot_product = sum([x * y for x, y in zip(a, b)])
        norm_a = sum([x ** 2 for x in a]) ** 0.5
        norm_b = sum([x ** 2 for x in b]) ** 0.5

        return dot_product / (norm_a * norm_b)
    
    def score(self, query: str, top_k: int = 3) -> dict:
        """
        Retrieval method that will embed and input query with an ollama embedding model.
        
        Args:
            query: that is embedded for search
            top_n: number of top n retrieved documents

        Returns:
            similarities: containing ids, document chunks, cosine similarities  
            with respect to the query
        """
        query_embedding = ollama.embed(model=self.embedding_model, input=query)["embeddings"][0]
        similarities = []

        for _, row in self.vector_db.iterrows():
            id = row["id"]
            chunk = row["chunk"]
            chunk_embedding = row[self.embedding_model]

            similarity = self.cosine_similarity(query_embedding, chunk_embedding)

            similarities.append((id, chunk, similarity))

        # Sorting by similarity in descending order
        top_similarities = sorted(similarities, key=lambda x: x[2], reverse=True)[:top_k]
        
        # Converting retrieved chunks to a dictionary
        similarities_dict = {
            "id": [id for id, _, _ in top_similarities],
            "chunk": [chunk for _, chunk, _ in top_similarities],
            "cosine_sim": [sim for _, _, sim in top_similarities]
        }

        return similarities_dict


if __name__ == "__main__":
    """
    Example usage of CosineEmbeddings.
    """
    
    EMBEDDING_MODEL = "nomic-embed-text"
    vector_db = pd.read_json("data/vector_db.jsonl", lines=True)
    test_question = "When was activity theory created?"
    top_k = 3

    retriever = CosineEmbeddings(EMBEDDING_MODEL, vector_db)
    retrieved_docs = retriever.score(test_question, top_k)

    print("\nRetrieved chunks:")
    for i, (chunk_id, chunk_text) in enumerate(zip(retrieved_docs["id"], retrieved_docs["chunk"])):
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(f"{chunk_text[0][:200]}..." )
        print("-" * 90)
    