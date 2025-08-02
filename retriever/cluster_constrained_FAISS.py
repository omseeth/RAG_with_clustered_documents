"""
cluster_constrained_FAISS.py

This retriever will use the Facebook AI Similarity Search (FAISS) library to retrieve documents.
"""
import faiss
faiss.omp_set_num_threads(1)  # Necessary for stability of faiss
import numpy as np
import pandas as pd
import ollama
import os
from sklearn.preprocessing import normalize
import sys
from typing import Tuple, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class ClusterConstrainedFAISS():
    """
    Clustering chunks with k-means, and using cluster-based FAISS indices for retrieval. The number of clusters to consider per retrieval and the number or retrieved documents can be set with top_k_clusters and top_k with the score() method.
    """
    def __init__(self, EMBEDDING_MODEL: str, vector_db: pd.DataFrame, max_clusters: int, top_k_clusters: int):
        self.embedding_model = EMBEDDING_MODEL
        self.vector_db = vector_db
        # FAISS requires numpy arrays
        # If normalized, we can use the embeddings for cosine similarity with faiss
        self.embeddings = normalize(
            np.vstack(self.vector_db[self.embedding_model].values).astype(np.float32),
            norm="l2", axis=1
        )
        self.d = self.get_embedding_dim()  # FAISS needs to know dimension size
        self.max_clusters = max_clusters  # Decide with silhouette coefficients
        self.top_k_clusters = top_k_clusters
        self.kmeans, self.cluster_assignments = self.kmeans_calc()
        self.cluster_to_index, self.cluster_to_chunks = self.faiss_indexing()
    
    def get_embedding_dim(self):
        """
        Helper method to determine the embedding dimension.
        """
        return len(self.vector_db[self.embedding_model][0])
    
    def kmeans_calc(self) -> Tuple[faiss.Kmeans, List[int]]:
        """
        FAISS based clustering with k-means.

        Returns:
            A list of assigned clusters for each embedding.
        """
        kmeans = faiss.Kmeans(d=self.d, k=self.max_clusters, niter=20, verbose=True)
        kmeans.train(self.embeddings)

        _, cluster_assignments = kmeans.index.search(self.embeddings, 1)
        cluster_assignments = cluster_assignments.flatten()

        return kmeans, cluster_assignments

    def faiss_indexing(self) -> Tuple[dict, dict]:
        """
        FAISS works with indices that will be used for the search of a query. We can use the FAISS inner product index for cosine similarity search, if the vector embeddings are normalized.

        Returns:
            A tuple of dictionaries: cluster_to_index assigns a FAISS index to each cluster, cluster_to_chunks contains the corresponding text chunks with their ids.
        """
        cluster_to_index = {}
        cluster_to_chunks = {}

        for cluster_id in range(self.max_clusters):

            chunks = self.vector_db["chunk"]
            idx = self.vector_db["id"]
            
            cluster_embeddings = self.embeddings[self.cluster_assignments == cluster_id]
            cluster_chunks = [(idx, doc) for idx, doc, cid in zip(idx, chunks, self.cluster_assignments) if cid == cluster_id]

            # Inner product results in cosine similarity if the vectors are normalized
            index = faiss.IndexFlatIP(self.d)
            index.add(cluster_embeddings)

            cluster_to_index[cluster_id] = index
            cluster_to_chunks[cluster_id] = cluster_chunks

        return cluster_to_index, cluster_to_chunks

    def score(self, query: str, top_k: int) -> dict:
        """
        The incoming query is embedded. This embedding is used to search for the top_k_clusters based on the distance between the query embedding and each cluster centroid. Finally, all relevant clusters are searched for the top_k results. A final list of results is recompiled based on top_k.

        Args:
            query, to search similar chunks in the database through FAISS   
                indices
            top_k, the number of top similar chunks to be retrieved.
        
        Returns:
            dict, containing the top similar chunks with their ids and the 
                cosine similarities w.r.s.t. the query
        """
        query_embedding = ollama.embed(model=self.embedding_model, input=query)["embeddings"][0]
        # Converting embedding to numpy, normalizing and reshaping it to two dimensions
        query_embedding = normalize(
            np.array(query_embedding, dtype=np.float32).reshape(1, -1),
            norm="l2", axis=1
        )
        
        # Distances to centroid of clusters n (closer is better)
        _, closest_clusters = self.kmeans.index.search(query_embedding, self.top_k_clusters)

        cluster_similarities = []

        # Look for top_k in each cluster 
        for cluster_id in closest_clusters[0]:

            index = self.cluster_to_index[cluster_id]
            chunks = self.cluster_to_chunks[cluster_id]

            similarities, I = index.search(query_embedding, top_k)
            retrieved_chunks = [chunks[i] for i in I[0]]

            for sim, chunk in zip(similarities[0], retrieved_chunks):
                cluster_similarities.append((chunk[0], chunk[1], float(sim)))
        
        # Sorting by similarity in descending order, compiling final top_k
        top_similarities = sorted(cluster_similarities, key=lambda x: x[2], reverse=True)[:top_k]
        
        # Converting retrieved chunks to a dictionary
        similarities_dict = {
            "id": [id for id, _, _ in top_similarities],
            "chunk": [chunk for _, chunk, _ in top_similarities],
            "cosine_sim": [sim for _, _, sim in top_similarities]
        }
        
        return similarities_dict

if __name__ == "__main__":
    """
    Example usage of ClusterConstrainedFAISS.
    """
    # We can work with vector_db as it also contains the chunks
    EMBEDDING_MODEL = "nomic-embed-text"
    vector_db = pd.read_json("data/vector_db.jsonl", lines=True)
    test_question = "When was activity theory created?"
    max_clusters = 4  # Amount of clusters for clustering with k-means
    top_k_clusters = 2  # Amount of clusters to consider for retrieval
    top_k_retrievals = 3 

    retriever = ClusterConstrainedFAISS(EMBEDDING_MODEL, vector_db, max_clusters, top_k_clusters)
    retrieved_docs = retriever.score(test_question, top_k_retrievals)

    # Output of retrieved chunks
    print("\nRetrieved chunks:")
    for i, (chunk_id, chunk_text) in enumerate(zip(retrieved_docs["id"], retrieved_docs["chunk"])):
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(f"{chunk_text[0][:200]}..." )
        print("-" * 90)
