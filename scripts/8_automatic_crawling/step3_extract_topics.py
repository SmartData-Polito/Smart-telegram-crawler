#!/usr/bin/env python3
"""
STEP 3: Extract topics from trained LDA model.
Usage: python step3_extract_topics.py --level 0 --topn 20
"""

import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import joblib

# ======================== CONFIGURATION ========================
SEED = 42
NUM_TOP_DOCS = 3
NUM_RANDOM_DOCS = 3
DEFAULT_TOP_WORDS = 20

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Extract topics from LDA model")
    parser.add_argument("--level", type=str, required=True, help="Hierarchy level")
    parser.add_argument("--topn", type=int, default=DEFAULT_TOP_WORDS, help="Top words per topic")
    args = parser.parse_args()
    
    level = args.level
    topn = args.topn
    log_time(f"Extracting topics for level {level}")
    
    # Paths
    base_dir = f"../results/levels_automatic/level_{level}"
    preprocess_dir = f"{base_dir}/preprocessing"
    lda_dir = f"{base_dir}/lda"
    
    model_path = f"{lda_dir}/models/lda_best.joblib"
    vectorizer_path = f"{lda_dir}/vectorizer.joblib"
    docs_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    best_k_path = f"{lda_dir}/best_k.json"
    
    # Outputs
    topics_txt_path = f"{lda_dir}/topics_keywords.txt"
    topics_json_path = f"{lda_dir}/topics_for_classification.json"
    
    # Load metadata
    if not os.path.exists(best_k_path):
        log_time(f"ERROR: Metadata not found: {best_k_path}")
        return
    
    with open(best_k_path, "r") as f:
        metadata = json.load(f)
    
    best_k = metadata["best_k"]
    log_time(f"Best K: {best_k}")
    
    # Load model
    if not os.path.exists(model_path):
        log_time(f"ERROR: Model not found: {model_path}")
        return
    
    log_time("Loading LDA model...")
    lda_model = joblib.load(model_path)
    
    # Load vectorizer
    log_time("Loading vectorizer...")
    vectorizer = joblib.load(vectorizer_path)
    vocabulary = vectorizer.get_feature_names_out()
    
    # Load documents for examples
    log_time("Loading documents...")
    df_docs = pd.read_csv(docs_path, sep='\t', compression='gzip', 
                          usecols=['text_llm', 'channel_id'])
    df_docs = df_docs[df_docs['text_llm'].astype(str).str.strip() != '']
    df_docs['text_llm'] = df_docs['text_llm'].astype(str)
    
    documents = df_docs['text_llm'].tolist()
    channel_ids = df_docs['channel_id'].tolist()
    num_docs = len(documents)
    log_time(f"Loaded {num_docs} documents")
    
    # Transform documents to get topic distributions
    log_time("Computing document-topic matrix...")
    
    # Need to use LDA preprocessing (text_lda column)
    df_lda = pd.read_csv(docs_path, sep='\t', compression='gzip', usecols=['text_lda'])
    df_lda = df_lda[df_lda['text_lda'].astype(str).str.strip() != '']
    docs_lda = df_lda['text_lda'].astype(str).tolist()
    
    X_docs = vectorizer.transform(docs_lda)
    doc_topic_matrix = lda_model.transform(X_docs)  # Shape: (n_docs, n_topics)
    
    # Get dominant topic per document
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    
    # Extract topics data
    log_time(f"Extracting top-{topn} words for {best_k} topics...")
    
    topic_word_matrix = lda_model.components_  # Shape: (n_topics, n_terms)
    rng = np.random.default_rng(SEED)
    
    topics_data = []  # For JSON output
    
    # Write human-readable output
    with open(topics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Level {level} - {best_k} topics - Top {topn} keywords\n")
        f.write("=" * 80 + "\n\n")
        
        for topic_idx in range(best_k):
            # Get top words
            top_indices = np.argsort(topic_word_matrix[topic_idx])[::-1][:topn]
            top_words = [vocabulary[i] for i in top_indices]
            
            # Store for JSON
            topics_data.append({
                "topic_id": topic_idx,
                "keywords": top_words[:10],  # Top 10 for classification
                "all_keywords": top_words
            })
            
            f.write(f"Topic {topic_idx}: {', '.join(top_words)}\n")
            
            # Get topic scores for all docs
            topic_scores = doc_topic_matrix[:, topic_idx]
            
            # Top documents (highest scores)
            top_doc_indices = np.argsort(topic_scores)[::-1][:NUM_TOP_DOCS]
            
            f.write(f"  Top {NUM_TOP_DOCS} documents:\n")
            for rank, doc_idx in enumerate(top_doc_indices, 1):
                score = topic_scores[doc_idx]
                snippet = documents[doc_idx][:200].replace('\n', ' ')
                f.write(f"    {rank}) [score={score:.3f}] {snippet}...\n")
            
            # Random documents where this topic is dominant
            candidates = np.where(dominant_topics == topic_idx)[0]
            candidates = np.setdiff1d(candidates, top_doc_indices)
            
            if len(candidates) == 0:
                candidates = np.setdiff1d(np.arange(num_docs), top_doc_indices)
            
            if len(candidates) > 0:
                n_sample = min(NUM_RANDOM_DOCS, len(candidates))
                random_indices = rng.choice(candidates, size=n_sample, replace=False)
                
                f.write(f"  {n_sample} random documents:\n")
                for rank, doc_idx in enumerate(random_indices, 1):
                    score = topic_scores[doc_idx]
                    snippet = documents[doc_idx][:200].replace('\n', ' ')
                    f.write(f"    {rank}) [score={score:.3f}] {snippet}...\n")
            
            f.write("\n")
    
    log_time(f"Saved topics to {topics_txt_path}")
    
    # Save JSON for classification
    with open(topics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "level": level,
            "num_topics": best_k,
            "topics": topics_data
        }, f, indent=2)
    
    log_time(f"Saved topics JSON to {topics_json_path}")
    
    # Also save document-topic assignments for later analysis
    assignments_path = f"{lda_dir}/doc_topic_assignments.csv.gz"
    
    df_assignments = pd.DataFrame({
        'channel_id': channel_ids,
        'dominant_topic': dominant_topics,
        'max_topic_score': np.max(doc_topic_matrix, axis=1)
    })
    df_assignments.to_csv(assignments_path, index=False, compression='gzip')
    log_time(f"Saved document assignments to {assignments_path}")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{lda_dir}/step3_completed.txt", "w") as f:
        f.write(f"Topic extraction completed in {total_time:.2f}s\n")
        f.write(f"Topics: {best_k}\n")
        f.write(f"Documents: {num_docs}\n")

if __name__ == "__main__":
    main()