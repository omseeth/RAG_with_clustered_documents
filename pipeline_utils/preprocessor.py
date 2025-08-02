"""
preprocessing.py

To avoid stop words and unfrequent words, further preprocessing steps are implemented with spacy.
"""

import json
import os
import pandas as pd
import sys
import spacy
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


class Preprocessor:
    """
    Preprocesses data with tokenization, punctuation, and stop word removal. Terms that appear only once in the dataset are removed, too. The result is being saved to a new .jsonl file.
    """

    def __init__(self, documents: pd.DataFrame, path: str = "data/tokenized_db.jsonl"):
        self.documents = documents
        self.chunks = self.documents["documents"].tolist()
        self.cleaned_chunks = []
        self.final_chunks = []
        self.nlp = spacy.load("en_core_web_sm")
        self.path = path
        self.term_count_dict = {}
        
        self.remove_stopwords()
        self.count_terms()
        self.remove_terms()
        self.save_final_chunks()

    def remove_stopwords(self):
        """
        Removes stopwords using spacy's tokenizer and stop word list. Also handles punctuation and lemmatization.
        """
        self.cleaned_chunks = []

        for chunk in tqdm(self.chunks, desc="Tokenizing documents", total=len(self.chunks)):
            text = chunk[0]
            doc = self.nlp(text)
            filtered = [
                token.text for token in doc
                if not token.is_stop and not token.is_punct and not token.is_space
            ]
            self.cleaned_chunks.append(filtered)

    def count_terms(self):
        """
        Counts terms from document chunks so that we can remove terms from documents that only appear once.
        """
        for chunk in self.cleaned_chunks:
            for token in chunk:
                word_count = self.term_count_dict.get(token, 0) + 1
                self.term_count_dict[token] = word_count
    
    def remove_terms(self):
        """
        Filters chunks from terms that appear only once in the overall document corpus.
        """
        self.final_chunks = [[token for token in text if self.term_count_dict[token] > 1] for text in self.cleaned_chunks]
    
    def save_final_chunks(self):
        """
        Writes ids, questions, unprocessed chunk, as well as tokenized and filtered chunk to self.path
        """
        i = 0

        with open(self.path, "a") as f:
        
            for row, tokenized_text in zip(self.documents.itertuples(index=False), self.final_chunks):
                idx = row.id
                question = row.question
                chunk = row.documents
                dataset_name = row.dataset_name
                
                json.dump({"id": i, "original_id": idx, "dataset_name": dataset_name, "question": question, "chunk": chunk, "tokenized_chunk": tokenized_text}, f)
                f.write("\n")
                i += 1


if __name__ == "__main__":
    """
    Example usage of Preprocessor.
    """

    df = pd.read_json("data/hagrid_full.jsonl", lines=True)
    preprocessor = Preprocessor(df)
