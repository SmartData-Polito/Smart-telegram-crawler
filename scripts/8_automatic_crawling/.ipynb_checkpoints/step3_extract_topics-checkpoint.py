#!/usr/bin/env python3
"""
STEP 3: Extract topics from trained LDA model.

Usage: python step3_extract_topics.py --level 0 --topn 60

Output: topics/

MODIFICHE:
- Aggiunto calcolo e salvataggio di doc_topic_matrix usando lda_model.inference() (batch, veloce)
- Salvato in lda/doc_topic_matrix.npy per riuso in step4 e step5
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
DEFAULT_TOP_WORDS = 60

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
    base_dir = f"../../results/levels_automatic/level_{level}"
    preprocess_dir = f"{base_dir}/preprocessing"
    lda_dir = f"{base_dir}/lda"
    topics_dir = f"{base_dir}/topics"
    os.makedirs(topics_dir, exist_ok=True)

    model_path = f"{lda_dir}/models/lda_best.joblib"
    dictionary_path = f"{lda_dir}/vectorizer.joblib"
    docs_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    best_k_path = f"{lda_dir}/best_k.json"
    
    # Output per doc_topic_matrix (riusata da step4 e step5)
    doc_topic_matrix_path = f"{lda_dir}/doc_topic_matrix_level_{level}.npy"

    # Outputs
    topics_txt_path = f"{topics_dir}/topics_keywords.txt"
    topics_json_path = f"{topics_dir}/topics_for_classification.json"

    if not os.path.exists(best_k_path):
        log_time(f"ERROR: Metadata not found: {best_k_path}")
        return

    with open(best_k_path, "r") as f:
        metadata = json.load(f)
    best_k = metadata["best_k"]
    log_time(f"Best K: {best_k}")

    if not os.path.exists(model_path):
        log_time(f"ERROR: Model not found: {model_path}")
        return

    log_time("Loading LDA model...")
    lda_model = joblib.load(model_path)

    log_time("Loading dictionary...")
    dictionary = joblib.load(dictionary_path)
    vocab_size = len(dictionary)
    vocabulary = np.array([dictionary[i] for i in range(vocab_size)])

    log_time("Loading documents...")
    df_docs = pd.read_csv(docs_path, sep='\t', compression='gzip', usecols=['text_llm', 'text_lda', 'channel_id'])
    df_docs = df_docs[df_docs['text_llm'].astype(str).str.strip() != '']
    df_docs = df_docs[df_docs['text_lda'].astype(str).str.strip() != '']
    df_docs['text_llm'] = df_docs['text_llm'].astype(str)
    df_docs['text_lda'] = df_docs['text_lda'].astype(str)
    documents = df_docs['text_llm'].tolist()
    docs_lda = df_docs['text_lda'].tolist()
    num_docs = len(documents)
    log_time(f"Loaded {num_docs} documents")

    log_time("Building corpus for Gensim...")
    tokenized_lda = [doc.split() for doc in docs_lda]
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_lda]

    log_time("Computing document-topic matrix (batch inference)...")
    doc_topic_matrix, _ = lda_model.inference(corpus)
    doc_topic_matrix = doc_topic_matrix / doc_topic_matrix.sum(axis=1, keepdims=True)
    
    log_time(f"Saving doc_topic_matrix to {doc_topic_matrix_path}...")
    np.save(doc_topic_matrix_path, doc_topic_matrix)
    log_time(f"doc_topic_matrix shape: {doc_topic_matrix.shape}")

    dominant_topics = np.argmax(doc_topic_matrix, axis=1)

    log_time(f"Extracting top-{topn} words for {best_k} topics...")
    rng = np.random.default_rng(SEED)

    topics_data = []
    with open(topics_txt_path, "w", encoding="utf-8") as f:
        f.write(f"Level {level} - {best_k} topics - Top {topn} keywords\n")
        f.write("=" * 80 + "\n\n")

        for topic_idx in range(best_k):
            top_words = [word for word, _ in lda_model.show_topic(topic_idx, topn=topn)]

            topics_data.append({
                "topic_id": topic_idx,
                "keywords": top_words[:10],
                "all_keywords": top_words
            })

            f.write(f"Topic {topic_idx}: {', '.join(top_words)}\n")

            topic_scores = doc_topic_matrix[:, topic_idx]
            top_doc_indices = np.argsort(topic_scores)[::-1][:NUM_TOP_DOCS]
            f.write(f"  Top {NUM_TOP_DOCS} documents:\n")
            for rank, doc_idx in enumerate(top_doc_indices, 1):
                score = topic_scores[doc_idx]
                snippet = documents[doc_idx][:200].replace('\n', ' ')
                f.write(f"    {rank}) [score={score:.3f}] {snippet}...\n")

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

    with open(topics_json_path, "w", encoding="utf-8") as f:
        json.dump({
            "level": level,
            "num_topics": best_k,
            "topics": topics_data
        }, f, indent=2)
    log_time(f"Saved topics JSON to {topics_json_path}")

    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")

    with open(f"{topics_dir}/step3_completed.txt", "w") as f:
        f.write(f"Topic extraction completed in {total_time:.2f}s\n")
        f.write(f"Topics: {best_k}\n")
        f.write(f"Documents: {num_docs}\n")
        f.write(f"doc_topic_matrix saved to: {doc_topic_matrix_path}\n")

if __name__ == "__main__":
    main()