#!/usr/bin/env python3
"""
STEP 2: Train LDA model with Optuna optimization.
Usage: python step2_lda_train.py --level 0
       python step2_lda_train.py --level 0 --base-dir ../../results/experiments/peak_jul_aug
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
import optuna
from gensim.corpora import Dictionary
from gensim.models import LdaModel, LdaMulticore
from gensim.models.coherencemodel import CoherenceModel

optuna.logging.set_verbosity(optuna.logging.WARNING)

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
SEED = 42
OPTUNA_TRIALS = 6
OPTUNA_FRACTION = 0.10      # 10% base
OPTUNA_MAX_DOCS = 50000     # Tetto massimo per tuning
OPTUNA_MIN_DOCS = 5000      # Minimo per tuning
COHERENCE_SUBSET = 5000
N_JOBS = max(4, os.cpu_count() - 1)

np.random.seed(SEED)

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    log_time(f"Starting LDA training for level {level}")
    log_time(f"  Base dir: {base_dir}")
    
    # Paths
    level_dir = f"{base_dir}/level_{level}"
    preprocess_dir = f"{level_dir}/preprocessing"
    lda_dir = f"{level_dir}/lda"
    
    input_file = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    
    # Clean previous outputs
    if os.path.exists(lda_dir):
        import shutil
        shutil.rmtree(lda_dir)
    os.makedirs(lda_dir, exist_ok=True)
    os.makedirs(f"{lda_dir}/models", exist_ok=True)
    
    log_time("Cleaned previous outputs")
    
    # Load data
    t_start = start_timer("load_data")
    log_time(f"Loading data from {input_file}")
    
    if not os.path.exists(input_file):
        log_time(f"ERROR: Input file not found: {input_file}")
        sys.exit(1)
    
    df = pd.read_csv(input_file, sep='\t', compression='gzip')
    
    if len(df) == 0:
        log_time("ERROR: No documents to process")
        sys.exit(1)
    
    documents = df['text_lda'].astype(str).tolist()
    end_timer("load_data", t_start)
    log_time(f"Loaded {len(documents):,} documents")
    
    # K range - FIXED LOGIC
    n_docs = len(documents)
    
    # Calculate bounds
    k_min_calc = max(10, int(np.sqrt(n_docs / 10)))  # More conservative lower bound
    k_max_calc = max(50, min(200, int(np.sqrt(n_docs) * 1.5)))  # Ensure at least 50
    
    # Ensure k_min <= k_max
    k_min = min(k_min_calc, k_max_calc)
    k_max = max(k_min_calc, k_max_calc)
    
    # Final safety check
    if k_min > k_max:
        k_min, k_max = k_max, k_min
    if k_min == k_max:
        k_max = k_min + 20
    
    log_time(f"K range: [{k_min}, {k_max}]")
    
    # LDA params
    passes = 4
    chunksize = max(100, n_docs // 50)
    log_time(f"LDA: passes={passes}, chunksize={chunksize}")
    log_time(f"Using {N_JOBS} workers")
    
    # Build dictionary
    t_start = start_timer("build_dictionary")
    log_time("Building dictionary...")
    tokenized_docs = [doc.split() for doc in documents]
    dictionary = Dictionary(tokenized_docs)
    dictionary.filter_extremes(no_below=5, no_above=0.5)
    end_timer("build_dictionary", t_start)
    
    # Build corpus
    t_start = start_timer("build_corpus")
    log_time("Building corpus...")
    corpus = [dictionary.doc2bow(doc) for doc in tokenized_docs]
    end_timer("build_corpus", t_start)
    
    log_time(f"Dictionary done | Vocab size: {len(dictionary):,}")
    
    # Save dictionary and corpus
    joblib.dump(dictionary, f"{lda_dir}/vectorizer.joblib")
    joblib.dump(corpus, f"{lda_dir}/corpus.joblib")
    
    # Optuna optimization
    t_start = start_timer("optuna_optimization")
    log_time("Starting Optuna optimization...")
    
    # Adaptive tuning size: 10% with min/max bounds
    n_tune_raw = int(len(documents) * OPTUNA_FRACTION)
    n_tune = min(OPTUNA_MAX_DOCS, max(OPTUNA_MIN_DOCS, n_tune_raw))
    
    tune_indices = np.random.choice(len(documents), min(n_tune, len(documents)), replace=False)
    tune_corpus = [corpus[i] for i in tune_indices]
    tune_docs = [tokenized_docs[i] for i in tune_indices]
    
    effective_pct = 100 * len(tune_corpus) / len(documents)
    log_time(f"Tuning on {len(tune_corpus):,}/{len(documents):,} documents ({effective_pct:.1f}%)")
    if n_tune_raw > OPTUNA_MAX_DOCS:
        log_time(f"  (capped from {n_tune_raw:,} to {OPTUNA_MAX_DOCS:,})")
    
    coherence_docs = tune_docs[:COHERENCE_SUBSET] if len(tune_docs) > COHERENCE_SUBSET else tune_docs
    
    def objective(trial):
        k = trial.suggest_int('num_topics', k_min, k_max)
        alpha = trial.suggest_categorical('alpha', ['symmetric', 'asymmetric'])
        eta = trial.suggest_categorical('eta', ['symmetric', 'auto'])
        
        model = LdaModel(
            corpus=tune_corpus,
            id2word=dictionary,
            num_topics=k,
            passes=2,
            chunksize=chunksize,
            alpha=alpha,
            eta=eta,
            random_state=SEED,
            per_word_topics=False
        )
        
        cm = CoherenceModel(
            model=model,
            texts=coherence_docs,
            dictionary=dictionary,
            coherence='c_npmi'
        )
        
        coherence = cm.get_coherence()
        
        # Fix per valori infiniti o NaN
        if not np.isfinite(coherence):
            coherence = -1.0  # Valore basso di default
        
        return coherence
    
    sampler = optuna.samplers.TPESampler(seed=SEED, multivariate=True)
    study = optuna.create_study(direction='maximize', sampler=sampler)
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=True)
    
    best_k = study.best_params['num_topics']
    best_alpha = study.best_params['alpha']
    best_eta = study.best_params['eta']
    best_coherence = study.best_value
    
    end_timer("optuna_optimization", t_start)
    log_time(f"Optuna done | Best k={best_k}, coherence={best_coherence:.4f}")
    
    # Train final model
    t_start = start_timer("train_final_model")
    log_time(f"Training final model with k={best_k} on full data...")
    
    final_model = LdaMulticore(
        corpus=corpus,
        id2word=dictionary,
        num_topics=best_k,
        passes=passes,
        chunksize=chunksize,
        alpha=best_alpha,
        eta=best_eta,
        random_state=SEED,
        workers=N_JOBS - 1,
        per_word_topics=False
    )
    
    end_timer("train_final_model", t_start)
    log_time("Training done")
    
    # Save model
    t_start = start_timer("save_model")
    model_path = f"{lda_dir}/models/lda_best.joblib"
    joblib.dump(final_model, model_path)
    end_timer("save_model", t_start)
    log_time(f"Model saved to {model_path}")
    
    # Save best params
    best_params = {
        "best_k": best_k,
        "best_alpha": best_alpha,
        "best_eta": best_eta,
        "best_coherence": best_coherence,
        "n_documents": len(documents),
        "n_tune_documents": len(tune_corpus),
        "tune_percentage": effective_pct,
        "vocab_size": len(dictionary),
        "k_range": [k_min, k_max]
    }
    
    with open(f"{lda_dir}/best_k.json", 'w') as f:
        json.dump(best_params, f, indent=2)
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{lda_dir}/step2_completed.txt", 'w') as f:
        f.write(f"Step 2: LDA Training\n")
        f.write(f"Level: {level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Documents: {len(documents):,}\n")
        f.write(f"  Tuning docs: {len(tune_corpus):,} ({effective_pct:.1f}%)\n")
        f.write(f"  Vocab size: {len(dictionary):,}\n")
        f.write(f"  K range: [{k_min}, {k_max}]\n")
        f.write(f"  Best K: {best_k}\n")
        f.write(f"  Best coherence: {best_coherence:.4f}\n")

if __name__ == "__main__":
    main()