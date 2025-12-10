#!/usr/bin/env python3
"""
STEP 3: Extract topics and compute document-topic matrix.
Usage: python step3_extract_topics.py --level 0
"""

import os
import sys
import time
import argparse
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib

# ======================== TIMING ========================
START_TIME = time.perf_counter()
STEP_TIMES = {}

def log_time(msg: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.2f}s] {msg}")

def start_timer(name: str) -> float:
    return time.perf_counter()

def end_timer(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    return elapsed

# ======================== CONFIG ========================
NUM_TOP_WORDS = 60
NUM_SAMPLE_DOCS = 5

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    args = parser.parse_args()
    
    level = args.level
    log_time(f"Extracting topics for level {level}")
    
    # Paths
    base_dir = f"../../results/levels_automatic/level_{level}"
    lda_dir = f"{base_dir}/lda"
    preprocess_dir = f"{base_dir}/preprocessing"
    topics_dir = f"{base_dir}/topics"
    os.makedirs(topics_dir, exist_ok=True)
    
    # Load best_k
    t_start = start_timer("load_config")
    with open(f"{lda_dir}/best_k.json", 'r') as f:
        best_params = json.load(f)
    best_k = best_params['best_k']
    end_timer("load_config", t_start)
    log_time(f"Best K: {best_k}")
    
    # Load model
    t_start = start_timer("load_model")
    log_time("Loading LDA model...")
    lda_model = joblib.load(f"{lda_dir}/models/lda_best.joblib")
    end_timer("load_model", t_start)
    
    # Load dictionary
    t_start = start_timer("load_dictionary")
    log_time("Loading dictionary...")
    dictionary = joblib.load(f"{lda_dir}/vectorizer.joblib")
    end_timer("load_dictionary", t_start)
    
    # Load documents
    t_start = start_timer("load_documents")
    log_time("Loading documents...")
    df = pd.read_csv(f"{preprocess_dir}/messages_english_clean.tsv.gz", 
                     sep='\t', compression='gzip')
    documents = df['text_lda'].astype(str).tolist()
    end_timer("load_documents", t_start)
    log_time(f"Loaded {len(documents)} documents")
    
    # Build corpus
    t_start = start_timer("build_corpus")
    log_time("Building corpus for Gensim...")
    tokenized_docs = [doc.split() for doc in documents]
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    end_timer("build_corpus", t_start)
    
    # Compute document-topic matrix
    t_start = start_timer("compute_doc_topic_matrix")
    log_time("Computing document-topic matrix (batch inference)...")
    
    gamma, _ = lda_model.inference(corpus)
    doc_topic_matrix = gamma / gamma.sum(axis=1, keepdims=True)
    
    end_timer("compute_doc_topic_matrix", t_start)
    
    # Save doc_topic_matrix
    t_start = start_timer("save_doc_topic_matrix")
    matrix_path = f"{lda_dir}/doc_topic_matrix_level_{level}.npy"
    log_time(f"Saving doc_topic_matrix to {matrix_path}...")
    np.save(matrix_path, doc_topic_matrix)
    end_timer("save_doc_topic_matrix", t_start)
    log_time(f"doc_topic_matrix shape: {doc_topic_matrix.shape}")
    
    # Extract top words for each topic
    t_start = start_timer("extract_keywords")
    log_time(f"Extracting top-{NUM_TOP_WORDS} words for {best_k} topics...")
    
    topics_data = []
    topics_text = []
    
    for topic_id in range(best_k):
        top_words = lda_model.show_topic(topic_id, topn=NUM_TOP_WORDS)
        keywords = [word for word, _ in top_words]
        
        # Find sample documents
        topic_probs = doc_topic_matrix[:, topic_id]
        top_doc_indices = np.argsort(topic_probs)[-NUM_SAMPLE_DOCS:][::-1]
        
        sample_docs = []
        for idx in top_doc_indices:
            if idx < len(df):
                sample_docs.append(df.iloc[idx]['text_llm'][:500] if 'text_llm' in df.columns else documents[idx][:500])
        
        topics_data.append({
            "topic_id": topic_id,
            "keywords": keywords,
            "sample_documents": sample_docs
        })
        
        topics_text.append(f"Topic {topic_id}: {', '.join(keywords[:20])}")
    
    end_timer("extract_keywords", t_start)
    
    # Save topics
    t_start = start_timer("save_topics")
    with open(f"{topics_dir}/topics_keywords.txt", 'w') as f:
        f.write('\n'.join(topics_text))
    log_time(f"Saved topics to {topics_dir}/topics_keywords.txt")
    
    with open(f"{topics_dir}/topics_for_classification.json", 'w') as f:
        json.dump(topics_data, f, indent=2)
    log_time(f"Saved topics JSON to {topics_dir}/topics_for_classification.json")
    end_timer("save_topics", t_start)
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{topics_dir}/step3_completed.txt", 'w') as f:
        f.write(f"Step 3: Topic Extraction\n")
        f.write(f"Level: {level}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Topics: {best_k}\n")
        f.write(f"  Documents: {len(documents)}\n")
        f.write(f"  Matrix shape: {doc_topic_matrix.shape}\n")

if __name__ == "__main__":
    main()