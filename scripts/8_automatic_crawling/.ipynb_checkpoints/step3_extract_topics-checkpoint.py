#!/usr/bin/env python3
"""
STEP 3: Extract topics and compute document-topic matrix.
Usage: python step3_extract_topics.py --level 0
       python step3_extract_topics.py --level 0 --base-dir ../../results/experiments/peak_jul_aug
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
MAX_FALLBACK_RATIO = 0.5  # Alert if more than 50% of sample docs need fallback

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    log_time(f"Extracting topics for level {level}")
    log_time(f"  Base dir: {base_dir}")
    
    # Paths
    level_dir = f"{base_dir}/level_{level}"
    lda_dir = f"{level_dir}/lda"
    preprocess_dir = f"{level_dir}/preprocessing"
    topics_dir = f"{level_dir}/topics"
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
    
    # Check text_llm quality BEFORE processing
    t_start = start_timer("check_text_llm_quality")
    if 'text_llm' in df.columns:
        total_rows = len(df)
        nan_count = df['text_llm'].isna().sum()
        non_string_count = (~df['text_llm'].apply(lambda x: isinstance(x, str) if pd.notna(x) else False)).sum()
        invalid_count = nan_count + (non_string_count - nan_count)
        
        fallback_ratio = invalid_count / total_rows if total_rows > 0 else 0
        
        log_time(f"text_llm quality check:")
        log_time(f"  Total rows: {total_rows}")
        log_time(f"  NaN values: {nan_count} ({100*nan_count/total_rows:.1f}%)")
        log_time(f"  Invalid (need fallback): {invalid_count} ({100*fallback_ratio:.1f}%)")
        
        if fallback_ratio > MAX_FALLBACK_RATIO:
            log_time(f"")
            log_time(f"{'='*60}")
            log_time(f"ALERT: Too many text_llm values need fallback!")
            log_time(f"  Fallback ratio: {100*fallback_ratio:.1f}% > {100*MAX_FALLBACK_RATIO:.0f}% threshold")
            log_time(f"  This indicates a data quality issue in preprocessing.")
            log_time(f"  Check step1_preprocess.py output.")
            log_time(f"{'='*60}")
            log_time(f"")
            sys.exit(1)
    else:
        log_time(f"WARNING: text_llm column not found, will use text_lda for samples")
    end_timer("check_text_llm_quality", t_start)
    
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
    
    topics_data = {
        "level": level,
        "num_topics": best_k,
        "topics": []
    }
    topics_text = []
    
    # Track fallbacks during sample extraction
    total_samples = 0
    fallback_samples = 0
    
    for topic_id in range(best_k):
        top_words = lda_model.show_topic(topic_id, topn=NUM_TOP_WORDS)
        keywords = [word for word, _ in top_words]
        
        # Find sample documents
        topic_probs = doc_topic_matrix[:, topic_id]
        top_doc_indices = np.argsort(topic_probs)[-NUM_SAMPLE_DOCS:][::-1]
        
        sample_docs = []
        for idx in top_doc_indices:
            total_samples += 1
            if idx < len(df):
                # Handle NaN values safely
                if 'text_llm' in df.columns:
                    text_val = df.iloc[idx]['text_llm']
                    if pd.notna(text_val) and isinstance(text_val, str):
                        sample_docs.append(text_val[:500])
                    else:
                        # Fallback to text_lda
                        fallback_samples += 1
                        sample_docs.append(documents[idx][:500])
                else:
                    sample_docs.append(documents[idx][:500])
        
        topics_data["topics"].append({
            "topic_id": topic_id,
            "keywords": keywords[:10],
            "all_keywords": keywords,
            "sample_documents": sample_docs
        })
        
        topics_text.append(f"Topic {topic_id}: {', '.join(keywords[:20])}")
    
    end_timer("extract_keywords", t_start)
    
    # Report fallback stats
    if total_samples > 0:
        sample_fallback_ratio = fallback_samples / total_samples
        log_time(f"Sample docs fallback: {fallback_samples}/{total_samples} ({100*sample_fallback_ratio:.1f}%)")
    
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
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Topics: {best_k}\n")
        f.write(f"  Documents: {len(documents)}\n")
        f.write(f"  Matrix shape: {doc_topic_matrix.shape}\n")
        f.write(f"  Sample docs fallback: {fallback_samples}/{total_samples}\n")

if __name__ == "__main__":
    main()