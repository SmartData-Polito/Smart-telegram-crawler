#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

"""
LDA (scikit-learn, variational EM) + Optuna (TPESampler) per ottimizzare SOLO k.
Nei trial (subset, default 10%) si massimizza c_v.
Alla fine si riallena su 100% e si calcolano tutte le metriche (c_v, c_npmi, diversity, balance).
Se si usa --tune_only, ora COMUNQUE salva modello e risultati finali.
"""

import time
import random
import argparse
import shutil
import joblib
import json
from datetime import datetime
import numpy as np
import pandas as pd

from contextlib import contextmanager
from sklearn.decomposition import LatentDirichletAllocation as SKL_LDA
from sklearn.feature_extraction.text import CountVectorizer

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from scipy.sparse import save_npz, load_npz

# === Optuna (TPE) ===
import optuna
from optuna.samplers import TPESampler

# ============================== LOGGER ===============================
START_TS = time.perf_counter()
_LAST_TS_BOX = [START_TS]

def p(message: str) -> None:
    now = time.perf_counter()
    total = now - START_TS
    delta = now - _LAST_TS_BOX[0]
    print(f"[{total:8.2f}s][+{delta:6.2f}s] {message}")
    _LAST_TS_BOX[0] = now

@contextmanager
def section(name: str):
    p(f"▶ START {name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        p(f"■ END   {name} ({time.perf_counter() - t0:.2f}s)")

# ============================== SEED/ARGS ===========================
SEED = 42
random.seed(SEED); np.random.seed(SEED)

parser = argparse.ArgumentParser(description="LDA + Optuna (TPE) su k, maximize c_v (subset) + train finale 100%")
parser.add_argument("--input", type=str, default="0")
parser.add_argument("--resume", action="store_true")
parser.add_argument("--optuna_frac", type=float, default=0.10, help="frazione di documenti per i trial (0<frac<=1)")
parser.add_argument(
    "--tune_only",
    action="store_true",
    help="Esegui solo l'ottimizzazione di k su subset (c_v) e NON allena/valuta su 100%. (MODIFICATO: salviamo comunque best model e risultati)"
)
args = parser.parse_args()
level_depth = args.input
p(f"level_depth={level_depth}")

# ============================== PATHS ===============================
input_path = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz"
base_root = f"../results/levels/level_{level_depth}/grid_search_lda/"
output_path_df_sampled = f"{base_root}df_sampled_level_{level_depth}.csv"
out_path_grid_search_results = f"{base_root}grid_search_results_level_{level_depth}.csv"
dir_models        = f"{base_root}lda_models_level_{level_depth}/"
dir_vectorizers   = f"{base_root}vectorizers_level_{level_depth}/"
dir_bow           = f"{base_root}bag_of_words_level_{level_depth}/"

os.makedirs(base_root, exist_ok=True)
os.makedirs(dir_models, exist_ok=True)
os.makedirs(dir_vectorizers, exist_ok=True)
os.makedirs(dir_bow, exist_ok=True)
p(f"#debug2 dirs ready: {dir_models}, {dir_vectorizers}, {dir_bow}")

# ============================== PULIZIA =============================
def empty_dir(directory: str) -> None:
    if os.path.isdir(directory):
        for name in os.listdir(directory):
            full = os.path.join(directory, name)
            if os.path.isfile(full) or os.path.islink(full):
                os.unlink(full)
            else:
                shutil.rmtree(full)

if not args.resume:
    empty_dir(dir_models); empty_dir(dir_vectorizers); empty_dir(dir_bow)
    for path_to_remove in [output_path_df_sampled, out_path_grid_search_results]:
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove); p(f"[INFO] Rimosso file {path_to_remove}")
    p("[INFO] Pulizia completata.")
else:
    p("[RESUME] Nessuna pulizia.")

# ============================== LETTURA =============================
with section("READ TSV"):
    df_pre = pd.read_csv(input_path, sep="\t", compression="gzip")
    df_pre = df_pre[df_pre['text_preprocessed'].astype(str).str.strip() != ""]
    p("input accepted head:\n" + str(df_pre.head()))
    p(f"len :{len(df_pre)}")

df_all = df_pre.sample(frac=1.0, random_state=SEED).copy()
df_all['text_preprocessed'] = df_all['text_preprocessed'].astype(str)
df_all.to_csv(output_path_df_sampled, index=False)
p(f"#debug6 df_sampled len={len(df_all)} saving to {output_path_df_sampled}")

tokens_full = [s.split() for s in df_all['text_preprocessed'].tolist()]
docs_full = df_all['text_preprocessed'].tolist()
num_docs = int(len(docs_full))

# ===================== IPERPARAMETRI ADATTIVI =======================
def choose_num_topics(n_docs: int) -> list[int]:
    """3 candidati centrati su sqrt(n/10), clamp 10..400."""
    k_base = int(np.clip(round(np.sqrt(max(n_docs, 1) / 10.0)), 10, 400))
    return sorted({
        max(10, int(round(k_base * 0.6))),
        k_base,
        min(400, int(round(k_base * 1.4)))
    })

def vectorizer_params(n_docs: int):
    min_df_docs = max(2, int(round(0.001 * n_docs)))
    max_df_frac = 0.95 if n_docs < 5000 else 0.90
    ngram_range = (1, 1) if n_docs < 5000 else (1, 2)
    max_features = None if n_docs < 30000 else 50000
    return min_df_docs, max_df_frac, ngram_range, max_features

def lda_params(n_docs: int):
    method = 'batch' if n_docs < 5000 else 'online'
    batch_size = max(64, min(512, n_docs // 50))
    max_iter = 100 if n_docs < 10000 else 50
    learning_decay = 0.7
    return method, batch_size, max_iter, learning_decay

k_candidates = choose_num_topics(num_docs)
min_df_docs, max_df_frac, ngram_range, max_features = vectorizer_params(num_docs)
lda_method, lda_batch_size, lda_max_iter, lda_decay = lda_params(num_docs)
p(f"[ADAPT] num_docs={num_docs} → K_LIST={k_candidates} | min_df={min_df_docs} max_df={max_df_frac} ngram_range={ngram_range} max_features={max_features}")
p(f"[ADAPT] LDA: method={lda_method} batch_size={lda_batch_size} max_iter={lda_max_iter} decay={lda_decay}")

# ========================= VECTORIZER ===============================
with section("VECTORIZER FIT"):
    vect_suffix = f"CV_ng{ngram_range[0]}-{ngram_range[1]}_minDF{min_df_docs}_maxDF{max_df_frac}_maxF{max_features or 'None'}"
    path_vectorizer = os.path.join(dir_vectorizers, f"vectorizer_{vect_suffix}.joblib")
    path_X = os.path.join(dir_bow,  f"X_{vect_suffix}.npz")

    if os.path.exists(path_vectorizer) and os.path.exists(path_X) and args.resume:
        vectorizer = joblib.load(path_vectorizer)
        X_full = load_npz(path_X)
        p("[RESUME] Vectorizer e matrice X caricati.")
    else:
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=min_df_docs,
            max_df=max_df_frac,
            ngram_range=ngram_range,
            max_features=max_features
        )
        X_full = vectorizer.fit_transform(docs_full)
        joblib.dump(vectorizer, path_vectorizer)
        save_npz(path_X, X_full)
        p(f"vectorizer saved → {path_vectorizer} | X saved → {path_X} | shape={X_full.shape}")

vocab = vectorizer.get_feature_names_out()

# ============================= METRICHE =============================
def top_words_from_topics(lda_model: SKL_LDA, vocabulary, topn: int = 10) -> list[list[str]]:
    """Prende le top parole per topic dalla matrice componenti (topic-word)."""
    W = lda_model.components_
    topics_words = []
    for t in range(W.shape[0]):
        idx = np.argsort(W[t])[::-1][:topn]
        topics_words.append([vocabulary[i] for i in idx])
    return topics_words

def compute_all_metrics(topics_words, tokenized_docs):
    """Metriche complete (usate solo sul modello finale 100%)."""
    diversity = float(TopicDiversity(topk=10).score({"topics": topics_words}))
    c_v       = float(Coherence(topk=10, texts=tokenized_docs, measure='c_v').score({"topics": topics_words}))
    c_npmi    = float(Coherence(topk=10, texts=tokenized_docs, measure='c_npmi').score({"topics": topics_words}))
    return diversity, c_v, c_npmi

def compute_cv_only(topics_words, tokenized_docs):
    """Solo c_v (per i trial Optuna, calcolata sul subset)."""
    return float(Coherence(topk=10, texts=tokenized_docs, measure='c_v').score({"topics": topics_words}))

def topic_balance(doc_topic_prob: np.ndarray):
    """Semplice misura di uniformità delle assegnazioni documento→topic."""
    assigned = doc_topic_prob.argmax(axis=1)
    counts = pd.Series(assigned).value_counts()
    if counts.empty:
        return 0.0, 0, 0, 0
    min_size, max_size = int(counts.min()), int(counts.max())
    balance = 1.0 - (max_size - min_size) / float(len(assigned))
    return float(np.clip(balance, 0.0, 1.0)), int(counts.nunique()), min_size, max_size

# ========== NORMALIZZAZIONE ROBUSTA ==========
def robust_minmax_series(values: pd.Series, q_low=0.10, q_high=0.90) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").astype("float64")
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), 0.5, dtype="float64"), index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).astype("float64")

def avg_score_robust(df: pd.DataFrame) -> pd.Series:
    parts, weights = [], []
    if 'coherence_cv' in df.columns:
        parts.append(robust_minmax_series(df['coherence_cv'])); weights.append(0.4)
    if 'diversity' in df.columns:
        parts.append(robust_minmax_series(df['diversity']));    weights.append(0.3)
    if 'coherence_npmi' in df.columns:
        npmi01 = ((pd.to_numeric(df['coherence_npmi'], errors="coerce").astype("float64") + 1.0) / 2.0).clip(0, 1)
        parts.append(robust_minmax_series(npmi01));              weights.append(0.2)
    if 'balance' in df.columns:
        parts.append(robust_minmax_series(df['balance']));       weights.append(0.1)
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    wsum = sum(weights); weights = [w/wsum for w in weights]
    stacked = np.vstack([s.fillna(0.0).to_numpy(dtype="float64") for s in parts])
    return pd.Series(np.average(stacked, axis=0, weights=weights), index=df.index, dtype="float64")

# ======================== CSV HEADER ================================
result_columns = [
    'k','learning_method','batch_size','max_iter','learning_decay',
    'coherence_cv','coherence_npmi','diversity','balance',
    'effective_topics','min_topic','max_topic','vocab_size','n_docs',
    'model_path','vectorizer_path','suffix',
    'avg_score_robust'
]

def enforce_result_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    float_cols = ['learning_decay','coherence_cv','coherence_npmi','diversity','balance','avg_score_robust']
    int_cols   = ['k','batch_size','max_iter','effective_topics','min_topic','max_topic','vocab_size','n_docs']
    str_cols   = ['learning_method','model_path','vectorizer_path','suffix']
    for c in float_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    for c in int_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")
    for c in str_cols:
        if c in df.columns: df[c] = df[c].astype("string")
    return df

if not os.path.exists(out_path_grid_search_results):
    pd.DataFrame(columns=result_columns).to_csv(out_path_grid_search_results, index=False)
    p(f"#debug8 created results file: {out_path_grid_search_results}")
else:
    p(f"#debug8 results file exists: {out_path_grid_search_results}")

# =========================== TRAIN + METRICHE =======================
def train_and_collect(k_topics: int) -> dict:
    """Fit su 100% e metriche complete (usata SOLO dopo l'ottimizzazione)."""
    suffix = (
        f"LDA_k{k_topics}_ng{ngram_range[0]}-{ngram_range[1]}_"
        f"minDF{min_df_docs}_maxDF{max_df_frac}_maxF{max_features or 'None'}_"
        f"decay{lda_decay}"
    )
    path_model = os.path.join(dir_models, f"{suffix}.joblib")
    path_vec   = os.path.join(dir_vectorizers, f"vectorizer_{suffix}.joblib")

    if os.path.exists(path_model) and args.resume:
        lda = joblib.load(path_model); p(f"[RESUME] loaded model {path_model}")
    else:
        lda = SKL_LDA(
            n_components=int(k_topics),
            learning_method=lda_method,
            batch_size=int(lda_batch_size),
            max_iter=int(lda_max_iter),
            learning_decay=float(lda_decay),
            random_state=SEED,
            evaluate_every=0,
            verbose=2,
            doc_topic_prior=None,
            topic_word_prior=None,
            n_jobs=1
        )
        t0 = time.perf_counter()
        lda.fit(X_full)
        p(f"fit LDA(K={k_topics}, decay={lda_decay}) in {time.perf_counter()-t0:.2f}s")
        joblib.dump(lda, path_model); joblib.dump(vectorizer, path_vec)

    topics_words = top_words_from_topics(lda, vocab, topn=10)
    diversity, c_v, c_npmi = compute_all_metrics(topics_words, tokens_full)
    
    doc_topic = lda.transform(X_full)
    balance, eff_k, min_topic_size, max_topic_size = topic_balance(doc_topic)

    return dict(
        k=int(k_topics),
        learning_method=str(lda_method),
        batch_size=int(lda_batch_size),
        max_iter=int(lda_max_iter),
        learning_decay=float(lda_decay),
        coherence_cv=float(c_v),
        coherence_npmi=float(c_npmi),
        diversity=float(diversity),
        balance=float(balance),
        effective_topics=int(eff_k),
        min_topic=int(min_topic_size),
        max_topic=int(max_topic_size),
        vocab_size=int(len(vocab)),
        n_docs=int(num_docs),
        model_path=path_model,
        vectorizer_path=path_vec,
        suffix=suffix
    )


# ============================ OPTIMIZATION ==========================
with section("OPTUNA (TPE) su k — maximize c_v (subset)"):
    tuning_start = time.perf_counter()
    
    # range di k
    k_min, k_max = max(10, min(k_candidates)), min(400, max(k_candidates))
    p(f"[OPTUNA] k range = [{k_min}, {k_max}]")

    # subset per i trial
    opt_frac = float(args.optuna_frac)
    rng = np.random.RandomState(SEED)
    n_tune = max(100, int(np.ceil(opt_frac * num_docs)))
    tune_idx = rng.choice(num_docs, size=n_tune, replace=False)
    X_tune = X_full[tune_idx]
    tokens_tune = [tokens_full[i] for i in tune_idx]
    p(f"[OPTUNA] tuning on {n_tune}/{num_docs} docs ({100*opt_frac:.1f}%)")

    def objective(trial: optuna.trial.Trial) -> float:
        """Valutiamo k massimizzando c_v sul subset."""
        k = trial.suggest_int("k", k_min, k_max, step=1)
        lda = SKL_LDA(
            n_components=int(k),
            learning_method=lda_method,
            batch_size=int(lda_batch_size),
            max_iter=int(lda_max_iter),
            learning_decay=float(lda_decay),
            random_state=SEED,
            evaluate_every=0,
            verbose=0,
            doc_topic_prior=None,
            topic_word_prior=None,
            n_jobs=8
        )
        lda.fit(X_tune)
        topics_words = top_words_from_topics(lda, vocab, topn=10)
        c_v = compute_cv_only(topics_words, tokens_tune)
        trial.set_user_attr("coh_cv", float(c_v))
        return float(c_v)

    sampler = TPESampler(seed=SEED, multivariate=True, n_startup_trials=5)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    n_trials = min(30, k_max - k_min + 1)
    p(f"[OPTUNA] running {n_trials} trials…")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best_k = int(study.best_trial.params["k"])
    best_cv = float(study.best_trial.user_attrs.get("coh_cv"))
    
    tuning_seconds = time.perf_counter() - tuning_start
    p(f"[OPTUNA] best k={best_k} | c_v(subset)={best_cv:.4f}")

    # log minimale dei trial
    trials_log = os.path.join(base_root, f"optuna_summary_k_{level_depth}.csv")
    pd.DataFrame(
        [{"trial": t.number, "k": t.params["k"], "coh_cv_subset": t.user_attrs.get("coh_cv"), "value": t.value}
         for t in study.trials]
    ).to_csv(trials_log, index=False)
    p(f"[OPTUNA] summary saved → {trials_log}")

    # Salva sempre metadati best-k leggeri
    best_k_info = {
        "level_depth": level_depth,
        "best_k": best_k,
        "c_v_subset": best_cv,
        "k_min": k_min,
        "k_max": k_max,
        "optuna_frac": opt_frac,
        "n_tune_docs": int(X_tune.shape[0]),
        "seed": SEED,
        "timestamp_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z"
    }
    best_k_path = os.path.join(base_root, f"optuna_best_k_level_{level_depth}.json")
    with open(best_k_path, "w", encoding="utf-8") as f:
        json.dump(best_k_info, f, ensure_ascii=False, indent=2)
    p(f"[OPTUNA] best-k metadata saved → {best_k_path}")

    # ============== MODIFICA: salva sempre modello e risultati ==========
    if args.tune_only:
        p("[TUNE-ONLY] Salvo comunque modello e riga risultati con best_k.")

    # train finale su 100% + metriche complete (sempre eseguito)
    training_start = time.perf_counter()
    row = train_and_collect(best_k)
    training_seconds = time.perf_counter() - training_start
    
    df_new = pd.DataFrame([row])
    df_new['avg_score_robust'] = avg_score_robust(df_new)

    if os.path.exists(out_path_grid_search_results):
        df_exist = pd.read_csv(out_path_grid_search_results)
        if set(result_columns).issubset(df_exist.columns):
            df_merged = pd.concat([df_exist[result_columns], df_new[result_columns]], ignore_index=True)
        else:
            df_merged = pd.concat([df_exist, df_new[result_columns]], ignore_index=True)
    else:
        df_merged = df_new[result_columns]

    df_merged = enforce_result_dtypes(df_merged)
    df_merged.to_csv(out_path_grid_search_results, index=False)
    p(f"[OPTUNA] appended best-k row; total rows = {len(df_merged)}")

# ======================== COMPLETION FLAG ===========================
total_seconds = time.perf_counter() - START_TS

# FORMATO MINIMALE RICHIESTO
minimal_text = f"""Total Time: {total_seconds:.2f}s
Tuning Time (10%): {tuning_seconds:.2f}s
Training Time (100%): {training_seconds:.2f}s
Best K: {best_k}
"""
with open(os.path.join(base_root, "grid_search_lda_completed_successfully.txt"), "w") as f:
    f.write(minimal_text)
p("#debug16 grid_search_lda_completed_successfully.txt written")