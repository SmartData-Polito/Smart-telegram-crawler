#!/usr/bin/env python3
"""
step2_lda_train.py
Train LDA model with Optuna hyperparameter tuning.
CORRETTO: Calcola coherence sui dati completi, non solo sul sample di tuning.
"""

import os
import sys
import json
import time
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import optuna
from gensim.corpora import Dictionary
from gensim.models import LdaMulticore, CoherenceModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ======================== CONFIGURATION ========================
MIN_K = 30
MAX_K_RATIO = 0.0007  # K_max = n_docs * ratio
TUNE_FRACTION = 0.10  # Use 10% for tuning (or max 50k)
MAX_TUNE_DOCS = 50000
N_TRIALS = 6
PASSES = 4
RANDOM_STATE = 42

def log(msg):
    elapsed = time.time() - START_TIME
    print(f"[{elapsed:8.2f}s] {msg}")

def clean_previous_outputs(base_dir, level):
    """Remove previous LDA outputs."""
    lda_dir = f"{base_dir}/level_{level}/lda"
    if os.path.exists(lda_dir):
        for item in os.listdir(lda_dir):
            if item not in ['models']:
                path = f"{lda_dir}/{item}"
                if os.path.isfile(path):
                    os.remove(path)
        models_dir = f"{lda_dir}/models"
        if os.path.exists(models_dir):
            for item in os.listdir(models_dir):
                os.remove(f"{models_dir}/{item}")

def compute_coherence_safe(model, texts, dictionary, per_topic=False):
    """
    Compute coherence safely, handling infinity/NaN.
    Returns mean of valid topic coherences if overall is infinite.
    """
    try:
        coherence_model = CoherenceModel(
            model=model,
            texts=texts,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        
        if per_topic:
            return coherence_model.get_coherence_per_topic()
        
        coherence = coherence_model.get_coherence()
        
        # Se infinito o NaN, calcola media dei topic validi
        if not np.isfinite(coherence):
            per_topic_coherence = coherence_model.get_coherence_per_topic()
            valid = [c for c in per_topic_coherence if np.isfinite(c)]
            if valid:
                coherence = np.mean(valid)
            else:
                coherence = -1.0
        
        return coherence
    except Exception as e:
        print(f"    [WARN] Coherence computation failed: {e}")
        return -1.0

def main():
    global START_TIME
    START_TIME = time.time()
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=int, required=True)
    parser.add_argument('--base-dir', type=str, required=True)
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    
    log(f"Starting LDA training for level {level}")
    log(f"  Base dir: {base_dir}")
    
    # Clean previous outputs
    clean_previous_outputs(base_dir, level)
    log("Cleaned previous outputs")
    
    # Create output directories
    lda_dir = f"{base_dir}/level_{level}/lda"
    models_dir = f"{lda_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Load data
    input_file = f"{base_dir}/level_{level}/preprocessing/messages_english_clean.tsv.gz"
    log(f"Loading data from {input_file}")
    
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    texts = [str(doc).split() for doc in df['text_lda'].fillna('')]
    n_docs = len(texts)
    
    log(f"Loaded {n_docs:,} documents")
    
    # Determine K range
    k_max = max(MIN_K + 10, int(n_docs * MAX_K_RATIO))
    k_min = MIN_K
    
    # Adjust for small datasets
    if n_docs < 10000:
        k_max = min(k_max, n_docs // 100)
        k_min = min(k_min, k_max - 10)
        k_min = max(10, k_min)
    
    log(f"K range: [{k_min}, {k_max}]")
    
    # Tuning parameters
    n_tune = min(int(n_docs * TUNE_FRACTION), MAX_TUNE_DOCS)
    if n_docs < MAX_TUNE_DOCS:
        n_tune = min(n_docs, int(n_docs * 0.55))  # Use more for small datasets
    
    chunksize = max(100, n_tune // PASSES // 2)
    workers = min(os.cpu_count() - 1, 71)
    
    log(f"LDA: passes={PASSES}, chunksize={chunksize}")
    log(f"Using {workers} workers")
    
    # Build dictionary
    log("Building dictionary...")
    dictionary = Dictionary(texts)
    dictionary.filter_extremes(no_below=5, no_above=0.5, keep_n=100000)
    
    # Build corpus
    log("Building corpus...")
    corpus = [dictionary.doc2bow(text) for text in texts]
    
    log(f"Dictionary done | Vocab size: {len(dictionary):,}")
    
    # Save dictionary and corpus
    dictionary.save(f"{models_dir}/dictionary.dict")
    joblib.dump(corpus, f"{lda_dir}/corpus.joblib")
    
    # Tuning corpus (sample)
    if n_docs > n_tune:
        np.random.seed(RANDOM_STATE)
        tune_indices = np.random.choice(n_docs, n_tune, replace=False)
        tune_corpus = [corpus[i] for i in tune_indices]
        tune_texts = [texts[i] for i in tune_indices]
    else:
        tune_corpus = corpus
        tune_texts = texts
        n_tune = n_docs
    
    tune_pct = 100 * n_tune / n_docs
    
    # Optuna optimization
    log("Starting Optuna optimization...")
    log(f"Tuning on {n_tune:,}/{n_docs:,} documents ({tune_pct:.1f}%)")
    
    def objective(trial):
        k = trial.suggest_int('k', k_min, k_max)
        alpha = trial.suggest_categorical('alpha', ['symmetric', 'asymmetric'])
        eta = trial.suggest_categorical('eta', ['symmetric', 'auto'])
        
        model = LdaMulticore(
            corpus=tune_corpus,
            id2word=dictionary,
            num_topics=k,
            passes=PASSES,
            chunksize=chunksize,
            random_state=RANDOM_STATE,
            workers=workers,
            alpha=alpha,
            eta=eta
        )
        
        coherence = compute_coherence_safe(model, tune_texts, dictionary)
        return coherence
    
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
    
    best_k = study.best_params['k']
    best_alpha = study.best_params['alpha']
    best_eta = study.best_params['eta']
    tuning_coherence = study.best_value
    
    log(f"Optuna done | Best k={best_k}, tuning coherence={tuning_coherence:.4f}")
    
    # Train final model on FULL data
    log(f"Training final model with k={best_k} on full data...")
    
    best_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        passes=PASSES,
        chunksize=max(100, n_docs // PASSES // 2),
        random_state=RANDOM_STATE,
        workers=workers,
        alpha=best_alpha,
        eta=best_eta
    )
    
    log("Training done")
    
    # ============================================================
    # NUOVO: Calcola coherence sui dati COMPLETI
    # ============================================================
    log("Computing final coherence on FULL data...")
    
    final_coherence = compute_coherence_safe(best_model, texts, dictionary)
    
    log(f"Final coherence (full data): {final_coherence:.4f}")
    
    # Calcola anche statistiche per topic
    per_topic_coherence = compute_coherence_safe(best_model, texts, dictionary, per_topic=True)
    
    if isinstance(per_topic_coherence, list):
        valid_coherences = [c for c in per_topic_coherence if np.isfinite(c)]
        n_valid = len(valid_coherences)
        n_infinite = sum(1 for c in per_topic_coherence if np.isinf(c))
        n_nan = sum(1 for c in per_topic_coherence if np.isnan(c))
        
        coherence_stats = {
            'total_topics': len(per_topic_coherence),
            'valid_topics': n_valid,
            'infinite_topics': n_infinite,
            'nan_topics': n_nan,
            'mean': float(np.mean(valid_coherences)) if valid_coherences else -1.0,
            'median': float(np.median(valid_coherences)) if valid_coherences else -1.0,
            'min': float(min(valid_coherences)) if valid_coherences else -1.0,
            'max': float(max(valid_coherences)) if valid_coherences else -1.0,
        }
        
        log(f"  Valid topics: {n_valid}/{len(per_topic_coherence)}")
        log(f"  Mean coherence: {coherence_stats['mean']:.4f}")
        log(f"  Median coherence: {coherence_stats['median']:.4f}")
    else:
        coherence_stats = None
    
    # Save model
    model_path = f"{models_dir}/lda_best.joblib"
    joblib.dump(best_model, model_path)
    log(f"Model saved to {model_path}")
    
    # Save info
    info = {
        'best_k': best_k,
        'best_alpha': best_alpha,
        'best_eta': best_eta,
        'tuning_coherence': tuning_coherence,  # Coherence durante tuning (sample)
        'best_coherence': final_coherence,      # Coherence finale (full data)
        'coherence_stats': coherence_stats,     # Statistiche dettagliate
        'n_documents': n_docs,
        'n_tune_documents': n_tune,
        'tune_percentage': tune_pct,
        'vocab_size': len(dictionary),
        'k_range': [k_min, k_max],
    }
    
    with open(f"{lda_dir}/best_k.json", 'w') as f:
        json.dump(info, f, indent=2)
    
    # Mark as completed
    with open(f"{lda_dir}/step2_completed.txt", 'w') as f:
        f.write(f"Completed at {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    log(f"COMPLETED in {time.time() - START_TIME:.2f}s")

if __name__ == "__main__":
    main()