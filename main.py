""" 
main.py 

This scripts instantiates all objects and starts Q&As with RAG.
"""

import ollama
import os
import pandas as pd
import spacy
import sys

from pipeline_utils.run_all_preprocessing import run_all_preprocessing
from retriever.cosine_embeddings import CosineEmbeddings
from retriever.BM25 import BM25
from retriever.cluster_constrained_retrieval import ClusterConstrainedRetrieval, BERTClassifier
from retriever.cluster_constrained_FAISS import ClusterConstrainedFAISS

def select_retriever():
    """
    Retriever selection
    """
    print("\nSelect retriever mode:")
    print("1. Cosine Similarity (nomic-embed-text)")
    print("2. Cosine Similarity (mxbai-embed-large)")
    print("3. BM25")
    print("4. Cluster Constrained Retrieval")
    print("5. FAISS")
    
    while True:
        choice = input("Enter your choice (1-6) [default: 1]: ").strip()
        if choice == "1":
            return "cosine_nomic"
        elif choice == "2":
            return "cosine_mxbai"
        elif choice == "3":
            return "bm25"
        elif choice == "4":
            return "cluster"
        elif choice == "5":
            return "faiss"
        else:
            print("Invalid choice. Please try again.")

def get_user_weights():
    """
    Get custom weights from user for each retriever
    """
    print("\nEnter weights for each retriever:")
    weights = []
    retriever_names = ["Cosine Similarity (nomic)", "Cosine Similarity (mxbai)", "BM25", "Cluster Constrained", "FAISS"]
    
    for name in retriever_names:
        while True:
            try:
                weight = float(input(f"Weight for {name} [default: 1.0]: ").strip() or "1.0")
                weights.append(weight)
                break
            except ValueError:
                print("Please enter a valid number.")
    
    print(f"Using weights: Cosine(nomic)={weights[0]}, Cosine(mxbai)={weights[1]}, BM25={weights[2]}, Cluster={weights[3]}, FAISS={weights[4]}")
    return weights

if __name__ == "__main__":
    LLM = "llama3:8b"
    top_k = 3  # Number of retrieved document chunks
    nlp = spacy.load("en_core_web_sm")

    ### Preparation of data, embeddings and preprocessing ###
    
    # Check if vector databases exist
    vector_db_path = "data/vector_db.jsonl"
    tokenized_db_path = "data/tokenized_db.jsonl"

    if os.path.exists(vector_db_path) and os.path.exists(tokenized_db_path):
        print("Found data/vector_db.jsonl and data/tokenized_db.jsonl. Warning: If clustering has been omitted then cluster based methods won't work.")
    else:
        run_all_preprocessing()

    ### Retrieval ###

    if os.path.exists(vector_db_path) and os.path.exists(tokenized_db_path):
        vector_db = pd.read_json(vector_db_path, lines=True)
        tokenized_db = pd.read_json(tokenized_db_path, lines=True)

        # Retrievers selection and initialization
        retriever_mode = select_retriever()
        
        # Set embedding model based on retriever mode
        if retriever_mode == "cosine_mxbai":
            EMBEDDING_MODEL = "mxbai-embed-large"
        else:
            EMBEDDING_MODEL = "nomic-embed-text"  # Default for other modes
    
        cosine_sim = CosineEmbeddings(EMBEDDING_MODEL, vector_db)
        
        bm25 = BM25(tokenized_db)
        
        # Initialize cluster constrained retrieval with MoBERT classifier
        cluster_retriever = None
        faiss_retriever = None
        mobert_classifier = None

        if retriever_mode in ["cluster"]:
            # Load MoBERT classifier
            model_path = "model/best_BERT_model.pth"
            if os.path.exists(model_path):
                mobert_classifier = BERTClassifier(vector_db, model_path)
                print("MoBERT classifier loaded.")
            else:
                print("You need to train a ModernBERT classifier based on clusters before you can continue with this option. Simply run: python pipeline_utils/bert_trainer.py --num_epochs <int>")
                sys.exit()
        
        # Initialize reranker for RRF mode
        reranker = None

        # Conversation loop
        while True:
            query = input("Ask a question: ")
            if query == "exit()":  # Stop conversation loop with exit()
                break
            
            # Preprocessing query with SpaCy
            tokenized_query = nlp(query)
            filtered_query = [
                token.text for token in tokenized_query
                if not token.is_stop and not token.is_punct and not token.is_space
            ]

            # Based on the retriever mode, retrieve the top_k chunks
            if retriever_mode == "cosine_nomic" or retriever_mode == "cosine_mxbai":
                # Use only embedding-based retrieval
                embedding_scores = cosine_sim.score(query, top_k)
                chunks_to_use = embedding_scores["chunk"]
                retrieved_ids = embedding_scores["id"]
                print(f"The {top_k} highest ranked chunks based on embeddings and cosine similarity are {retrieved_ids} in descending order.")

            elif retriever_mode == "bm25":
                # Use only BM25 retrieval
                bm25_scores = bm25.search(filtered_query, top_k)
                chunks_to_use = bm25_scores["chunk"]
                retrieved_ids = bm25_scores["id"]
                print(f"The {top_k} highest ranked chunks based on BM25 are {retrieved_ids} in descending order.")

            elif retriever_mode == "cluster":
                # Use MoBERT to predict relevant clusters, then retrieve from those clusters

                predicted_clusters = mobert_classifier.assign_cluster(query, top_k=3)
                cluster_retriever = ClusterConstrainedRetrieval(EMBEDDING_MODEL, vector_db)
                cluster_retriever.choose_from_cluster(predicted_clusters)
                
                cluster_scores = cluster_retriever.score(query, top_k)
                chunks_to_use = cluster_scores["chunk"]
                retrieved_ids = cluster_scores["id"]
                print(f"The {top_k} highest ranked chunks based on cluster constrained retrieval are {retrieved_ids} in descending order.")
            
            elif retriever_mode == "faiss":
                # Use FAISS to predict top clusters based on cluster centroids
                faiss_retriever = ClusterConstrainedFAISS(EMBEDDING_MODEL, vector_db, max_clusters=4, top_k_clusters=3)
          
                faiss_scores = faiss_retriever.score(query, top_k)
                chunks_to_use = faiss_scores["chunk"]
                retrieved_ids = faiss_scores["id"]
                print(f"The {top_k} highest ranked chunks based on FAISS cluster constrained retrieval are {retrieved_ids} in descending order.")

            ### Generation ###

            # Prepare chunks for prompt
            chunk_text = ""
            for i, chunk in enumerate(chunks_to_use):
                chunk_text += f"[{i+1}] {chunk}\n\n"

            instruction_prompt = f"""You are a helpful chatbot.

            ## Instruction
            Use only the following pieces of information to answer the question. Don't make up any new information.

            {chunk_text}

            ## Lacking information
            If there is no plausible information, simply state something like: "I cannot answer this because my information resources are limited".

            For example: What is existentialism?
            Answer: I cannot answer this because my information resources are limited.

            For example: Who is Angela Merkel?
            Answer: I cannot answer this because my information resources are limited.

            For example: When was Rolex founded?
            Answer: Rolex SA (formerly Wilsdorf and Davis) was originally founded in 1905 by Hans Wilsdorf and Alfred Davis in London. The company registered "Rolex" as the brand name of its watches in 1908.
            """

            response = ollama.chat(
                model=LLM,
                messages=[
                    {"role": "system", "content": instruction_prompt},
                    {"role": "user", "content": query},
                ],
                stream=False,
            )

            # Print the complete response
            print("\n")
            print(response["message"]["content"])
            print("\n")
