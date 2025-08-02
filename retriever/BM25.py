"""
BM25.py

The BM25 class has profited from code taken from https://ethen8181.github.io/machine-learning/search/bm25_intro.html.
"""

from typing import List
import math
import os
import pandas as pd
import spacy
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class BM25:
    """ 
    Retriever based on BM25.
    """

    def __init__(self, documents: pd.DataFrame, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.chunks = self.documents["tokenized_chunk"].tolist()
        self.b = b
        self.k1 = k1
        self.tf = []
        self.df = {}
        self.idf = {}
        self.doc_len = []
        self.corpus_size = 0
        self.avg_doc_len = 0

        self.fit()

    def fit(self):
        """
        Computes all necessary document statistics for BM25.

        - tf for each document
        - idf for each term
        - average document length
        - size n of corpus
        """
        for document in self.chunks:
            self.corpus_size += 1
            self.doc_len.append(len(document))

            # Compute tf (term frequency) per document
            frequencies = {}
            for term in document:
                term_count = frequencies.get(term, 0) + 1
                frequencies[term] = term_count

            self.tf.append(frequencies)

            # Compute df (document frequency) per term
            for term, _ in frequencies.items():
                df_count = self.df.get(term, 0) + 1
                self.df[term] = df_count

        for term, freq in self.df.items():
            self.idf[term] = math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
        
        self.avg_doc_len = sum(self.doc_len) / self.corpus_size
    
    def search(self, query: List[str], top_k: int = 3) -> dict:
        """
        Searches for a given query the documents with the highest BM25 score.

        Args:
            query: list of tokens
            top_k: number of top results from search

        Returns:
            top_scores: containing document id, original text, and BM25 score for query
        """
        scores = [self.score(query, index) for index in range(self.corpus_size)]
        unsorted_results = []

        for score, idx, doc in zip(scores, self.documents["id"].tolist(), self.documents["chunk"].tolist()):
            unsorted_results.append((idx, doc, score))
        
        top_scores = sorted(unsorted_results, key=lambda x: x[2], reverse=True)[:top_k]

        # Converting retrieved chunks to a dictionary
        scores_dict = {
            "id": [id for id, _, _ in top_scores],
            "chunk": [chunk for _, chunk, _ in top_scores],
            "cosine_sim": [sim for _, _, sim in top_scores]
        }

        return scores_dict

    def score(self, query: List[str], index: int) -> float:
        """
        Calculates BM25 score for a query with respect to a given document (referenced here by its index from the document statistics implemented with self.fit())

        Args:
            query: list of tokens
            index: reference to document for which the score is being calculated
        
        Returns:
            score: for query document pair
        """
        score = 0.0

        doc_len = self.doc_len[index]
        frequencies = self.tf[index]

        for term in query:
            if term not in frequencies:
                continue
            freq = frequencies[term]

            # BM25 score
            numerator = self.idf[term] * freq * (self.k1 + 1)
            denominator = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += (numerator / denominator)

        return score


if __name__ == "__main__":
    """
    Example usage of BM25.
    """
    tokenized_db = pd.read_json("data/tokenized_db.jsonl", lines=True)
    nlp = spacy.load("en_core_web_sm")
    query = "Who is the drummer for the band Superheist?"
    top_k = 3

    # Preprocessing query with SpaCy
    query = nlp(query)
    filtered_query = [
        token.text for token in query
        if not token.is_stop and not token.is_punct and not token.is_space
    ]

    # Loading BM25 and intializing document statistics
    bm25 = BM25(tokenized_db)

    bm25_scores = bm25.search(filtered_query, top_k)

    print("\nRetrieved chunks:")
    for i, (chunk_id, chunk_text) in enumerate(zip(bm25_scores["id"], bm25_scores["chunk"])):
        print(f"\nChunk {i+1} (ID: {chunk_id}):")
        print(f"{chunk_text[0][:200]}..." )
        print("-" * 90)