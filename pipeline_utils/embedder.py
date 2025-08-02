"""
embedder.py

This class contains an embedding template which will embed chunks with an Ollama model. The embeddings can be written to a jsonl file. Run script with:

$--embedding_model <model_name>

Example usage:

$python embedder.py --embedding_model mxbai-embed-large
"""

import argparse
import json
import ollama
import os
import pandas as pd
import sys
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

def parse_args():
    """
    Collecting arguments to embedd document chunks.
    """
    parser = argparse.ArgumentParser(
        description="Embedding data."
    )

    parser.add_argument("--embedding_model", type=str, 
    help="Name of embedding model to be used with Ollama.")

    return parser.parse_args()


class Embedder:
    """
    Embedding class for given documents stored as a pandas dataframe which rows containing. Each row in the vector DB stores:
        - "id": internal ID
        - "original_id": source document ID
        - "question": the question string
        - "chunk": the document chunk
        - "{model_name}": embedding for each model
    """
    
    def __init__(self, embedding_model: str, documents: pd.DataFrame, path: str = "data/vector_db.jsonl"):
        self.embedding_model = embedding_model
        self.documents = documents
        self.vector_db_path = path
        self.vector_db = self.load_vector_db()
        self.embed_documents()

    def load_vector_db(self):
        """
        Loads existing vector DB and returns a dict mapping original_id to data.
        """
        if not os.path.exists(self.vector_db_path):
            return {}

        vector_db = {}
        with open(self.vector_db_path, "r") as f:
            for line in f:
                entry = json.loads(line)
                vector_db[entry["original_id"]] = entry
        return vector_db

    def embed_documents(self):
        """
        Embeds documents, updating or adding entries with the new embedding model. Writes the complete vector DB back to the file.
        """
        for i, row in enumerate(tqdm(self.documents.itertuples(index=False), desc="Embedding documents", total=len(self.documents))):
            idx = row.id
            question = row.question
            chunk = row.documents
            dataset_name = row.dataset_name

            if idx not in self.vector_db:
                # New entry
                self.vector_db[idx] = {
                    "id": i,
                    "original_id": idx,
                    "dataset_name": dataset_name,
                    "question": question,
                    "chunk": chunk,
                }

            # Adding or updating embedding for the given model
            if self.embedding_model not in self.vector_db[idx]:
                embedding = ollama.embed(model=self.embedding_model, input=chunk[0])["embeddings"][0]
                self.vector_db[idx][self.embedding_model] = embedding

        self.write_vector_db()

    def write_vector_db(self):
        """
        Writes the entire vector DB back to the JSONL file.
        """
        with open(self.vector_db_path, "w") as f:
            for entry in self.vector_db.values():
                json.dump(entry, f)
                f.write("\n")    

if __name__ == "__main__":
    """
    Example usage of Embedder.
    """
    args = parse_args()
    EMBEDDING_MODEL = args.embedding_model

    if EMBEDDING_MODEL is None:
        EMBEDDING_MODEL = "nomic-embed-text"

    df = pd.read_json("data/hagrid_full.jsonl", lines=True)
    embedder = Embedder(EMBEDDING_MODEL, df)
    