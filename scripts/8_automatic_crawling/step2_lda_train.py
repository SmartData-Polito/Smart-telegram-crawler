#!/usr/bin/env python3
"""
STEP 2: Train LDA model with Optuna optimization.

Usage: python step2_lda_train.py --level 0

Output: lda/
"""

import os
import time
import random
import argparse
import json
import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore
from multiprocessing import cpu_count
from octis.evaluation_metrics.coherence_metrics import Coherence

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ======================== CONFIGURATION ========================
SEED = 42
try:
    N_JOBS = max(4, cpu_count() - 1)
except NotImplementedError:
    N_JOBS = 1

OPTUNA_FRACTION = 0.10
OPTUNA_TRIALS = 6

random.seed(SEED)
np.random.seed(SEED)

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== ADAPTIVE HYPERPARAMETERS ========================
def compute_num_topics_range(num_docs: int) -> tuple:
    k_base = int(np.clip(round(np.sqrt(max(num_docs, 1) / 10.0)), 10, 400))
    k_min = max(10, int(round(k_base * 0.6)))
    k_max = min(400, int(round(k_base * 1.4)))
    return k_min, k_max

def compute_lda_params(num_docs: int) -> dict:
    return {
        'passes': 4,
        'chunksize': max(2000, min(2500, num_docs // 50)),
        'decay': 0.5
    }

# ======================== COHERENCE METRIC ========================
def compute_coherence_cv(lda_model, dictionary, tokenized_docs, topn: int = 10) -> float:
    try:
        #from octis.evaluation_metrics.coherence_metrics import Coherence
        topic_words = []
        for topic_idx in range(lda_model.num_topics):
            top_words = [word for word, _ in lda_model.show_topic(topic_idx, topn=topn)]
            topic_words.append(top_words)
        coherence = Coherence(topk=topn, texts=tokenized_docs, measure='c_v')
        return float(coherence.score({"topics": topic_words}))
    except Exception as e:
        log_time(f"Warning: Coherence computation failed: {e}")
        return 0.0

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Train LDA with Optuna")
    parser.add_argument("--level", type=str, default="0", help="Hierarchy level")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    level = args.level
    log_time(f"Starting LDA training for level {level}")

    # Paths
    base_dir = f"../../results/levels_automatic/level_{level}"
    preprocess_dir = f"{base_dir}/preprocessing"
    lda_dir = f"{base_dir}/lda"
    input_file_messages_english_clean = f"{preprocess_dir}/messages_english_clean.tsv.gz"

    os.makedirs(lda_dir, exist_ok=True)
    models_dir = f"{lda_dir}/models"
    os.makedirs(models_dir, exist_ok=True)

    dictionary_path = f"{lda_dir}/vectorizer.joblib"
    corpus_path = f"{lda_dir}/corpus.joblib"
    best_model_path = f"{models_dir}/lda_best.joblib"
    best_k_path = f"{lda_dir}/best_k.json"

    if not args.resume:
        for f in [dictionary_path, corpus_path, best_model_path, best_k_path]:
            if os.path.exists(f):
                os.remove(f)
        log_time("Cleaned previous outputs")

    log_time(f"Loading data from {input_file_messages_english_clean}")
    df = pd.read_csv(input_file_messages_english_clean, sep='\t', compression='gzip')
    df = df[df['text_lda'].astype(str).str.strip() != '']
    documents = df['text_lda'].astype(str).tolist()
    tokenized_docs = [doc.split() for doc in documents]
    num_docs = len(documents)
    log_time(f"Loaded {num_docs} documents")

    k_min, k_max = compute_num_topics_range(num_docs)
    """
    lda_params = {
        'passes': x,
        'chunksize': x,
        'decay': x
    }   
    """
    lda_params = compute_lda_params(num_docs)

    log_time(f"K range: [{k_min}, {k_max}]")
    log_time(f"LDA: passes={lda_params['passes']}, chunksize={lda_params['chunksize']}")

    dictionary_start = time.perf_counter()
    if args.resume and os.path.exists(dictionary_path) and os.path.exists(corpus_path):
        log_time("Loading existing dictionary and corpus")
        dictionary = joblib.load(dictionary_path)
        corpus_full = joblib.load(corpus_path)
    else:
        log_time("Building dictionary...")
        dictionary = Dictionary(tokenized_docs)
        dictionary.filter_extremes(no_below=max(2, int(round(0.001 * num_docs))), 
                                   no_above=0.95 if num_docs < 5000 else 0.90)
        corpus_full = [dictionary.doc2bow(doc) for doc in tokenized_docs]
        joblib.dump(dictionary, dictionary_path)
        joblib.dump(corpus_full, corpus_path)

    vocab_size = len(dictionary)
    log_time(f"Dictionary done in {time.perf_counter() - dictionary_start:.2f}s | Vocab size: {vocab_size}")

    optuna_start = time.perf_counter()
    log_time("Starting Optuna optimization...")

    rng = np.random.RandomState(SEED)
    n_tune = max(100, int(np.ceil(OPTUNA_FRACTION * num_docs)))
    tune_indices = rng.choice(num_docs, size=n_tune, replace=False)
    corpus_tune = [corpus_full[i] for i in tune_indices]
    tokens_tune = [tokenized_docs[i] for i in tune_indices]

    log_time(f"Tuning on {n_tune}/{num_docs} documents ({OPTUNA_FRACTION*100:.0f}%)")

    def objective(trial):
        k = trial.suggest_int("k", k_min, k_max, step=1)
        lda = LdaMulticore(
            corpus=corpus_tune,
            id2word=dictionary,
            num_topics=k,
            passes=lda_params['passes'],
            chunksize=lda_params['chunksize'],
            decay=lda_params['decay'],
            random_state=SEED,
            workers=N_JOBS
        )
        cv_score = compute_coherence_cv(lda, dictionary, tokens_tune)
        trial.set_user_attr("cv_score", cv_score)
        return cv_score

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = TPESampler(seed=SEED, multivariate=True, n_startup_trials=3)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    n_trials = min(OPTUNA_TRIALS, k_max - k_min + 1)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_k = study.best_trial.params["k"]
    best_cv = study.best_trial.user_attrs.get("cv_score", 0.0)
    optuna_time = time.perf_counter() - optuna_start
    log_time(f"Optuna done in {optuna_time:.2f}s | Best k={best_k}, cv={best_cv:.4f}")

    trials_df = pd.DataFrame([
        {"trial": t.number, "k": t.params["k"], "cv_score": t.user_attrs.get("cv_score")}
        for t in study.trials
    ])
    trials_df.to_csv(f"{lda_dir}/optuna_trials.csv", index=False)

    train_start = time.perf_counter()
    log_time(f"Training final model with k={best_k} on full data...")

    final_lda = LdaMulticore(
        corpus=corpus_full,
        id2word=dictionary,
        num_topics=best_k,
        passes=lda_params['passes'],
        chunksize=lda_params['chunksize'],
        decay=lda_params['decay'],
        random_state=SEED,
        workers=N_JOBS
    )

    train_time = time.perf_counter() - train_start
    log_time(f"Training done in {train_time:.2f}s")

    joblib.dump(final_lda, best_model_path)
    log_time(f"Model saved to {best_model_path}")

    metadata = {
        "level": level,
        "best_k": best_k,
        "cv_score_subset": best_cv,
        "k_min": k_min,
        "k_max": k_max,
        "num_docs": num_docs,
        "vocab_size": vocab_size,
        "optuna_fraction": OPTUNA_FRACTION,
        "optuna_time_seconds": optuna_time,
        "training_time_seconds": train_time
    }
    with open(best_k_path, "w") as f:
        json.dump(metadata, f, indent=2)

    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")

    with open(f"{lda_dir}/step2_completed.txt", "w") as f:
        f.write(f"LDA training completed in {total_time:.2f}s\n")
        f.write(f"Best k: {best_k}\n")
        f.write(f"CV score (subset): {best_cv:.4f}\n")
        f.write(f"Optuna time: {optuna_time:.2f}s\n")
        f.write(f"Training time: {train_time:.2f}s\n")

if __name__ == "__main__":
    main()