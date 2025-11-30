#!/usr/bin/env python3
"""
STEP 2: Train LDA model with Optuna optimization.
Usage: python step2_lda_train.py --level 0
"""

import os
import time
import random
import argparse
import json
import shutil

import numpy as np
import pandas as pd
import joblib
import optuna
from optuna.samplers import TPESampler
from scipy.sparse import save_npz, load_npz
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Limit threads for stability
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

# ======================== CONFIGURATION ========================
SEED = 42
N_JOBS = 4  # Parallel jobs for LDA
OPTUNA_FRACTION = 0.10  # Use 10% of data for hyperparameter tuning
OPTUNA_TRIALS = 6  # Number of Optuna trials

random.seed(SEED)
np.random.seed(SEED)

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== ADAPTIVE HYPERPARAMETERS ========================
def compute_num_topics_range(num_docs: int) -> tuple:
    """Compute k range based on corpus size."""
    k_base = int(np.clip(round(np.sqrt(max(num_docs, 1) / 10.0)), 10, 400))
    k_min = max(10, int(round(k_base * 0.6)))
    k_max = min(400, int(round(k_base * 1.4)))
    return k_min, k_max

def compute_vectorizer_params(num_docs: int) -> dict:
    """Compute vectorizer parameters based on corpus size."""
    return {
        'min_df': max(2, int(round(0.001 * num_docs))),
        'max_df': 0.95 if num_docs < 5000 else 0.90,
        'ngram_range': (1, 1),
        'max_features': None if num_docs < 30000 else 50000
    }

def compute_lda_params(num_docs: int) -> dict:
    """Compute LDA parameters based on corpus size."""
    return {
        'method': 'batch' if num_docs < 5000 else 'online',
        'batch_size': max(64, min(512, num_docs // 50)),
        'max_iter': 50 if num_docs < 10000 else 25,
        'learning_decay': 0.7
    }

# ======================== COHERENCE METRIC ========================
def compute_coherence_cv(lda_model, vocabulary, tokenized_docs, topn: int = 10) -> float:
    """Compute c_v coherence score."""
    try:
        from octis.evaluation_metrics.coherence_metrics import Coherence
        
        # Extract top words per topic
        topic_words = []
        for topic_idx in range(lda_model.n_components):
            top_indices = np.argsort(lda_model.components_[topic_idx])[::-1][:topn]
            topic_words.append([vocabulary[i] for i in top_indices])
        
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
    base_dir = f"../results/levels_automatic/level_{level}"
    preprocess_dir = f"{base_dir}/preprocessing"
    lda_dir = f"{base_dir}/lda"
    
    input_file = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    
    # Create output directories
    os.makedirs(lda_dir, exist_ok=True)
    models_dir = f"{lda_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Output paths
    vectorizer_path = f"{lda_dir}/vectorizer.joblib"
    bow_matrix_path = f"{lda_dir}/bow_matrix.npz"
    best_model_path = f"{models_dir}/lda_best.joblib"
    best_k_path = f"{lda_dir}/best_k.json"
    
    # Clean previous run if not resuming
    if not args.resume:
        for f in [vectorizer_path, bow_matrix_path, best_model_path, best_k_path]:
            if os.path.exists(f):
                os.remove(f)
        log_time("Cleaned previous outputs")
    
    # Load preprocessed data
    log_time(f"Loading data from {input_file}")
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    df = df[df['text_lda'].astype(str).str.strip() != '']
    
    documents = df['text_lda'].astype(str).tolist()
    tokenized_docs = [doc.split() for doc in documents]
    num_docs = len(documents)
    
    log_time(f"Loaded {num_docs} documents")
    
    # Compute adaptive parameters
    k_min, k_max = compute_num_topics_range(num_docs)
    vec_params = compute_vectorizer_params(num_docs)
    lda_params = compute_lda_params(num_docs)
    
    log_time(f"K range: [{k_min}, {k_max}]")
    log_time(f"Vectorizer: min_df={vec_params['min_df']}, max_df={vec_params['max_df']}")
    log_time(f"LDA: method={lda_params['method']}, max_iter={lda_params['max_iter']}")
    
    # Step 1: Vectorize
    vectorizer_start = time.perf_counter()
    
    if args.resume and os.path.exists(vectorizer_path) and os.path.exists(bow_matrix_path):
        log_time("Loading existing vectorizer and BoW matrix")
        vectorizer = joblib.load(vectorizer_path)
        X_full = load_npz(bow_matrix_path)
    else:
        log_time("Fitting vectorizer...")
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=vec_params['min_df'],
            max_df=vec_params['max_df'],
            ngram_range=vec_params['ngram_range'],
            max_features=vec_params['max_features']
        )
        X_full = vectorizer.fit_transform(documents)
        joblib.dump(vectorizer, vectorizer_path)
        save_npz(bow_matrix_path, X_full)
    
    vocabulary = vectorizer.get_feature_names_out()
    log_time(f"Vectorizer done in {time.perf_counter() - vectorizer_start:.2f}s | Vocab size: {len(vocabulary)}")
    
    # Step 2: Optuna tuning on subset
    optuna_start = time.perf_counter()
    log_time("Starting Optuna optimization...")
    
    # Create tuning subset
    rng = np.random.RandomState(SEED)
    n_tune = max(100, int(np.ceil(OPTUNA_FRACTION * num_docs)))
    tune_indices = rng.choice(num_docs, size=n_tune, replace=False)
    X_tune = X_full[tune_indices]
    tokens_tune = [tokenized_docs[i] for i in tune_indices]
    
    log_time(f"Tuning on {n_tune}/{num_docs} documents ({OPTUNA_FRACTION*100:.0f}%)")
    
    def objective(trial):
        k = trial.suggest_int("k", k_min, k_max, step=1)
        
        lda = LatentDirichletAllocation(
            n_components=k,
            learning_method=lda_params['method'],
            batch_size=lda_params['batch_size'],
            max_iter=lda_params['max_iter'],
            learning_decay=lda_params['learning_decay'],
            random_state=SEED,
            n_jobs=N_JOBS,
            verbose=0
        )
        lda.fit(X_tune)
        
        cv_score = compute_coherence_cv(lda, vocabulary, tokens_tune)
        trial.set_user_attr("cv_score", cv_score)
        return cv_score
    
    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    sampler = TPESampler(seed=SEED, multivariate=True, n_startup_trials=3)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    
    n_trials = min(OPTUNA_TRIALS, k_max - k_min + 1)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    
    best_k = study.best_trial.params["k"]
    best_cv = study.best_trial.user_attrs.get("cv_score", 0.0)
    
    optuna_time = time.perf_counter() - optuna_start
    log_time(f"Optuna done in {optuna_time:.2f}s | Best k={best_k}, cv={best_cv:.4f}")
    
    # Save Optuna results
    trials_df = pd.DataFrame([
        {"trial": t.number, "k": t.params["k"], "cv_score": t.user_attrs.get("cv_score")}
        for t in study.trials
    ])
    trials_df.to_csv(f"{lda_dir}/optuna_trials.csv", index=False)
    
    # Step 3: Train final model on full data
    train_start = time.perf_counter()
    log_time(f"Training final model with k={best_k} on full data...")
    
    final_lda = LatentDirichletAllocation(
        n_components=best_k,
        learning_method=lda_params['method'],
        batch_size=lda_params['batch_size'],
        max_iter=lda_params['max_iter'],
        learning_decay=lda_params['learning_decay'],
        random_state=SEED,
        n_jobs=N_JOBS,
        verbose=1
    )
    final_lda.fit(X_full)
    
    train_time = time.perf_counter() - train_start
    log_time(f"Training done in {train_time:.2f}s")
    
    # Save model
    joblib.dump(final_lda, best_model_path)
    log_time(f"Model saved to {best_model_path}")
    
    # Save metadata
    metadata = {
        "level": level,
        "best_k": best_k,
        "cv_score_subset": best_cv,
        "k_min": k_min,
        "k_max": k_max,
        "num_docs": num_docs,
        "vocab_size": len(vocabulary),
        "optuna_fraction": OPTUNA_FRACTION,
        "optuna_time_seconds": optuna_time,
        "training_time_seconds": train_time
    }
    
    with open(best_k_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    # Write completion flag
    with open(f"{lda_dir}/step2_completed.txt", "w") as f:
        f.write(f"LDA training completed in {total_time:.2f}s\n")
        f.write(f"Best k: {best_k}\n")
        f.write(f"CV score (subset): {best_cv:.4f}\n")
        f.write(f"Optuna time: {optuna_time:.2f}s\n")
        f.write(f"Training time: {train_time:.2f}s\n")

if __name__ == "__main__":
    main()