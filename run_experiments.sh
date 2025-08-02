#!/bin/bash

# Reproducibility Script for Experiments

## Setup libraries ##
echo "Installing dependencies with poetry..."
poetry install
poetry run python -m spacy download en_core_web_sm

echo "Please, make sure Ollama is installed."

## Loading models ##
ollama pull nomic-embed-text
ollama pull mxbai-embed-large
ollama pull llama3:8b

echo "Starting Ollama server in the background..."
ollama serve > ollama.log 2>&1 &
sleep 3  # Give server a few seconds to start

## Preprocessing ##
echo "Step 1: Running preprocessing with nomic-embed-text and 4 clusters..."
python pipeline_utils/run_all_preprocessing.py --embedding_model nomic-embed-text --n_clusters 4

echo "Step 2: Running embedder with mxbai-embed-large..."
python pipeline_utils/embedder.py --embedding_model mxbai-embed-large

## Model Training ##
echo "Step 3: Training BERT with 3 epochs..."
python pipeline_utils/bert_trainer.py --num_epochs 3

## Experiments based on clustering ##
echo "Comparing retrieval quality with Success@k, MRR, and latency"

echo "Step 4: Evaluating retriever using BM25 with k=1..."
python evaluation/retriever_eval.py --retriever bm25 --k 1

echo "Step 4: Evaluating retriever using BM25 with k=3..."
python evaluation/retriever_eval.py --retriever bm25 --k 3

echo "Step 4: Evaluating retriever using BM25 with k=5..."
python evaluation/retriever_eval.py --retriever bm25 --k 5

echo "Step 4: Evaluating retriever using nomic-embed-text with k=1..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model nomic-embed-text --k 1

echo "Step 4: Evaluating retriever using nomic-embed-text with k=3..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model nomic-embed-text --k 3

echo "Step 4: Evaluating retriever using nomic-embed-text with k=5..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model nomic-embed-text --k 5

echo "Step 4: Evaluating retriever using nomic-embed-text with k=5..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model nomic-embed-text --k 5

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=1..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 1 --top_clusters 3

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=3..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 3 --top_clusters 3

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=5..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 5 --top_clusters 3

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from FAISS k=1..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 1 --top_clusters 3 --max_clusters 4

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from FAISS k=3..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 3 --top_clusters 3 --max_clusters 4 

echo "Step 4: Evaluating retriever using nomic-embed-text with clustering from FAISS k=5..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 5 --top_clusters 3 --max_clusters 4

echo "Step 4: Evaluating retriever using mxbai-embed-large with k=1..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model mxbai-embed-large --k 1

echo "Step 4: Evaluating retriever using mxbai-embed-large with k=3..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model mxbai-embed-large --k 3

echo "Step 4: Evaluating retriever using mxbai-embed-large with k=5..."
python evaluation/retriever_eval.py --retriever cosine --embedding_model mxbai-embed-large --k 5

echo "Comparing quality of ModernBERT based clustering for retrieval with nomic-embed-text"

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=1 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 1 --top_clusters 3

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=3 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 3 --top_clusters 3

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=5 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 5 --top_clusters 3

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=1 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 1 --top_clusters 2

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=3 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 3 --top_clusters 2

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=5 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 5 --top_clusters 2

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=1 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 1 --top_clusters 1

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=3 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 3 --top_clusters 1

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from ModernBERT k=5 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever MoBERT --embedding_model nomic-embed-text --k 5 --top_clusters 1

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=1 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 1 --top_clusters 3 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=3 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 3 --top_clusters 3 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=5 and considering top 3 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 5 --top_clusters 3 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=1 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 1 --top_clusters 2 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=3 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 3 --top_clusters 2 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=5 and considering top 2 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 5 --top_clusters 2 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=1 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 1 --top_clusters 1 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=3 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 3 --top_clusters 1 --max_clusters 4

echo "Step 5: Evaluating retriever using nomic-embed-text with clustering from FAISS k=5 and considering top 1 clusters..."
python evaluation/retriever_eval.py --retriever faiss --embedding_model nomic-embed-text --k 5 --top_clusters 1 --max_clusters 4