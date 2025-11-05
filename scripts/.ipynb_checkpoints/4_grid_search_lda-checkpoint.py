#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_MAX_THREADS"] = "1"

"""
here we use scikit learn with EM
istead of the gensim version with gibbs sampling
"""

import time
import random
import argparse
import shutil
import joblib
import numpy as np
import pandas as pd

from contextlib import contextmanager
from joblib import Parallel, delayed

from sklearn.decomposition import LatentDirichletAllocation as SKL_LDA
from sklearn.feature_extraction.text import CountVectorizer

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from scipy.sparse import save_npz, load_npz

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

parser = argparse.ArgumentParser(description="LDA grid search, CSV minimale e tipato")
parser.add_argument("--input", type=str, default="0")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()
level_depth = args.input
p(f"level_depth={level_depth}")

# ============================== PATHS ===============================
input_path = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz"
base_root = f"../results/levels/level_{level_depth}/grid_search_lda/"
output_path_df_sampled = f"{base_root}df_sampled_level_{level_depth}.csv"
out_path_grid_search_results = f"{base_root}grid_search_results_level_{level_depth}.csv"
dir_models      = f"{base_root}lda_models_level_{level_depth}/"
dir_vectorizers = f"{base_root}vectorizers_level_{level_depth}/"
dir_embeddings  = f"{base_root}embeddings_level_{level_depth}/"

os.makedirs(base_root, exist_ok=True)
os.makedirs(dir_models, exist_ok=True)
os.makedirs(dir_vectorizers, exist_ok=True)
os.makedirs(dir_embeddings, exist_ok=True)
p(f"#debug2 dirs ready: {dir_models}, {dir_vectorizers}, {dir_embeddings}")

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
    empty_dir(dir_models); empty_dir(dir_vectorizers); empty_dir(dir_embeddings)
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

df_sampled = df_pre.sample(frac=1.0, random_state=SEED).copy()
df_sampled['text_preprocessed'] = df_sampled['text_preprocessed'].astype(str)
df_sampled.to_csv(output_path_df_sampled, index=False)
p(f"#debug6 df_sampled len={len(df_sampled)} saving to {output_path_df_sampled}")

tokenized_texts = [s.split() for s in df_sampled['text_preprocessed'].tolist()]
raw_docs = df_sampled['text_preprocessed'].tolist()
num_docs = int(len(raw_docs))

# ===================== IPERPARAMETRI ADATTIVI =======================

def choose_num_topics(n_docs: int) -> list[int]:
    """
    Scelta automatica di 2–3 valori di 'num_topics' (k) per l'LDA in base al numero di documenti.

    Logica:
    - Più documenti ⇒ più temi servono per descrivere il corpus.
    - Usa la radice quadrata per far crescere k lentamente, evitando esplosioni su dataset enormi.
    - Applica limiti min/max: minimo 10, massimo 400.
    - Restituisce tre valori: uno base, uno minore (~0.6x) e uno maggiore (~1.4x).

    Esempio:
      n_docs = 10_000
      √(10000/10) = √1000 ≈ 31.6 → base = 32
      → valori restituiti = [19, 32, 45]
    """
    k_base = int(np.clip(round(np.sqrt(max(n_docs, 1) / 10.0)), 10, 400))
    return sorted({
        max(10, int(round(k_base * 0.6))),   # valore più piccolo (dataset più generico)
        k_base,                              # valore medio di riferimento
        min(400, int(round(k_base * 1.4)))   # valore più grande (temi più granulari)
    })


def compute_adaptive_vectorizer_params(n_docs: int):
    """
    Calcola i parametri per il CountVectorizer in base alla dimensione del dataset.

    Parametri:
    ----------
    n_docs : int
        Numero totale di documenti nel dataset (ad es. 2_000, 10_000, 50_000...).

    Output:
    -------
    Restituisce una tupla:
        (min_df_docs, max_df_fraction, ngram_range, max_features)

    Spiegazione:
    -------------
     min_df_docs  → filtro per parole troppo rare
        - Numero minimo di documenti in cui una parola deve comparire.
        - Evita termini casuali o con errori di battitura.
        - Formula:  min_df_docs = max(2, round(0.001 * n_docs))
        Esempio: n_docs = 20_000 → 0.001 * 20000 = 20 → min_df_docs = 20

     max_df_fraction  → filtro per parole troppo comuni
        - Parole presenti in troppi documenti sono inutili (stopword generalizzate).
        - 0.95 se dataset < 5000 docs, 0.90 altrimenti.
        Esempio: n_docs = 20_000 → max_df_fraction = 0.90

     ngram_range  → tipo di combinazioni di parole
        - (1,1) → solo singole parole
        - (1,2) → anche coppie (es. “white house”)
        - Dataset grandi → (1,2), piccoli → (1,1)

     max_features  → massimo numero di termini nel vocabolario
        - None si significa nessun limite se dataset piccolo (<30k)
        - 50.000 se dataset grande (>30k)
    """
    min_df_docs = max(2, int(round(0.001 * n_docs)))
    max_df_fraction = 0.95 if n_docs < 5000 else 0.90
    ngram_range = (1, 1) if n_docs < 5000 else (1, 2)
    max_features = None if n_docs < 30000 else 50000
    return min_df_docs, max_df_fraction, ngram_range, max_features

def compute_adaptive_lda_params(n_docs: int):
    """
    Parametri di addestramento adattivi per LDA (Latent Dirichlet Allocation).

    - method: 'batch' per dataset piccoli (<5k), 'online' per dataset grandi → 
      'online' aggiorna incrementale, riduce memoria.

    - batch_size: quanti documenti processare per volta.
      Calcolato come n_docs / 50, ma limitato tra 64 e 512.
      Esempio: 20_000 docs → batch_size = 20_000/50 = 400

    - max_iter: iterazioni massime per convergenza:
      100 iterazioni per dataset piccoli (più stabilità), 50 per grandi (performance).
      poichè si usa la versione scikit-learn di lda viene usato variation EM al poto di gibbs sampling

    - decay_candidates: possibili valori per il learning rate (quanto il modello "dimentica" il passato).
      0.5 = apprendimento più stabile, 0.7 = più reattivo ai nuovi batch.

    Restituisce tuple: (method, batch_size, max_iter, decay_candidates)
    """
    method = 'batch' if n_docs < 5000 else 'online'
    batch_size = max(64, min(512, n_docs // 50))
    max_iter = 100 if n_docs < 10000 else 50
    decay_candidates = [0.5, 0.7]
    return method, batch_size, max_iter, decay_candidates


candidate_num_topics = choose_num_topics(num_docs)
min_df_docs, max_df_fraction, ngram_range, max_features = compute_adaptive_vectorizer_params(num_docs)
lda_learning_method, lda_batch_size, lda_max_iter, lda_learning_decay_list = compute_adaptive_lda_params(num_docs)
p(f"[ADAPT] num_docs={num_docs} → K_LIST={candidate_num_topics} | min_df={min_df_docs} max_df={max_df_fraction} ngram_range={ngram_range} max_features={max_features}")
p(f"[ADAPT] LDA: method={lda_learning_method} batch_size={lda_batch_size} max_iter={lda_max_iter} decay_list={lda_learning_decay_list}")

# ========================= VECTORIZER ===============================
with section("VECTORIZER FIT"):
    vect_suffix = f"CV_ng{ngram_range[0]}-{ngram_range[1]}_minDF{min_df_docs}_maxDF{max_df_fraction}_maxF{max_features or 'None'}"
    vec_path = os.path.join(dir_vectorizers, f"vectorizer_{vect_suffix}.joblib")
    X_path   = os.path.join(dir_embeddings,  f"X_{vect_suffix}.npz")

    if os.path.exists(vec_path) and os.path.exists(X_path) and args.resume:
        vectorizer = joblib.load(vec_path)
        X = load_npz(X_path)  # sparse counts
        p("[RESUME] Vectorizer e matrice X caricati.")
    else:
        vectorizer = CountVectorizer(
            lowercase=False,
            token_pattern=r"(?u)\b\w+\b",
            min_df=min_df_docs,
            max_df=max_df_fraction,
            ngram_range=ngram_range,
            max_features=max_features
        )
        X = vectorizer.fit_transform(raw_docs)
        joblib.dump(vectorizer, vec_path)
        save_npz(X_path, X)
        p(f"vectorizer saved → {vec_path} | X saved → {X_path} | shape={X.shape}")

vocab = vectorizer.get_feature_names_out()
GLOBAL_X, GLOBAL_VOCAB, GLOBAL_TOKS = X, vocab, tokenized_texts

# ============================= METRICHE =============================
def extract_topics_top_words(lda_model: SKL_LDA, vocabulary, topn: int = 10) -> list[list[str]]:
    W = lda_model.components_  # [k, |V|] float64
    topics_words = []
    for t in range(W.shape[0]):
        idx = np.argsort(W[t])[::-1][:topn]
        topics_words.append([vocabulary[i] for i in idx])
    return topics_words

def compute_topic_metrics(topics_words, tokenized_docs):
    diversity = float(TopicDiversity(topk=10).score({"topics": topics_words}))
    coh_cv    = float(Coherence(topk=10, texts=tokenized_docs, measure='c_v').score({"topics": topics_words}))
    coh_npmi  = float(Coherence(topk=10, texts=tokenized_docs, measure='c_npmi').score({"topics": topics_words}))
    return diversity, coh_cv, coh_npmi

def compute_topic_balance(doc_topic_prob: np.ndarray):
    assigned = doc_topic_prob.argmax(axis=1)
    counts = pd.Series(assigned).value_counts()
    if counts.empty:
        return 0.0, 0, 0, 0
    min_size, max_size = int(counts.min()), int(counts.max())
    balance = 1.0 - (max_size - min_size) / float(len(assigned))
    return float(np.clip(balance, 0.0, 1.0)), int(counts.nunique()), min_size, max_size

# ========== NORMALIZZAZIONE ROBUSTA (IN RAM, NESSUNA COLONNA EXTRA) ==========
def robust_minmax_series(values: pd.Series, q_low=0.10, q_high=0.90) -> pd.Series:
    s = pd.to_numeric(values, errors="coerce").astype("float64")
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.full(len(s), 0.5, dtype=np.float64), index=s.index)
    return ((s - lo) / (hi - lo)).clip(0, 1).astype("float64")

def compute_avg_score_robust_in_memory(df: pd.DataFrame) -> pd.Series:
    """Restituisce SOLO la Serie 'avg_score_robust' senza aggiungere colonne."""
    parts, weights = [], []

    if 'coherence_cv' in df.columns:
        parts.append(robust_minmax_series(df['coherence_cv'])); weights.append(0.4)
    if 'diversity' in df.columns:
        parts.append(robust_minmax_series(df['diversity']));    weights.append(0.3)
    if 'coherence_npmi' in df.columns:
        # mappa [-1,1]→[0,1] senza creare colonne
        npmi01 = ((pd.to_numeric(df['coherence_npmi'], errors="coerce").astype("float64") + 1.0) / 2.0).clip(0, 1)
        parts.append(robust_minmax_series(npmi01));              weights.append(0.2)
    if 'balance' in df.columns:
        parts.append(robust_minmax_series(df['balance']));       weights.append(0.1)

    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    wsum = sum(weights); weights = [w/wsum for w in weights]
    stacked = np.vstack([s.fillna(0.0).to_numpy(dtype="float64") for s in parts])
    avg = np.average(stacked, axis=0, weights=weights)
    return pd.Series(avg, index=df.index, dtype="float64")

# ============================== GRID ================================
grid_configurations = []
for k_topics in candidate_num_topics:
    for decay in lda_learning_decay_list:
        grid_configurations.append(dict(
            n_components=int(k_topics),
            learning_method=str(lda_learning_method),
            batch_size=int(lda_batch_size),
            max_iter=int(lda_max_iter),
            learning_decay=float(decay)
        ))
p(f"Grid size = {len(grid_configurations)}")

# ======================== CSV HEADER (MINIMALE) =====================
result_columns = [
    'k','learning_method','batch_size','max_iter','learning_decay',      # iperparametri
    'coherence_cv','coherence_npmi','diversity','balance',               # metriche
    'effective_topics','min_topic','max_topic','vocab_size','n_docs',    # conteggi
    'model_path','vectorizer_path','suffix',                             # label
    'avg_score_robust'                                                   # metrica aggregata
]

def enforce_result_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Applica tipi standard SOLO alle colonne presenti."""
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

# =========================== SINGLE RUN =============================
def execute_single_configuration(cfg: dict) -> dict:
    X_matrix = GLOBAL_X; vocabulary = GLOBAL_VOCAB; tokenized_docs = GLOBAL_TOKS
    k_topics = int(cfg['n_components'])
    suffix = (
        f"LDA_k{k_topics}_ng{ngram_range[0]}-{ngram_range[1]}_"
        f"minDF{min_df_docs}_maxDF{max_df_fraction}_maxF{max_features or 'None'}_"
        f"decay{cfg['learning_decay']}"
    )
    model_path = os.path.join(dir_models, f"{suffix}.joblib")
    vec_path   = os.path.join(dir_vectorizers, f"vectorizer_{suffix}.joblib")

    if os.path.exists(model_path) and args.resume:
        lda_model = joblib.load(model_path); p(f"[RESUME] loaded model {model_path}")
    else:
        p("\nsingle execute_single_configuration call")
        lda_model = SKL_LDA(
            n_components=k_topics,
            learning_method=cfg['learning_method'],
            batch_size=int(cfg['batch_size']),
            max_iter=int(cfg['max_iter']),
            learning_decay=float(cfg['learning_decay']),
            random_state=SEED,
            evaluate_every=0,
            verbose=2,
            doc_topic_prior=None,
            topic_word_prior=None,
            n_jobs=1
        )
        t0 = time.perf_counter()
        lda_model.fit(X_matrix)
        p(f"fit LDA(K={k_topics}, decay={cfg['learning_decay']}) in {time.perf_counter()-t0:.2f}s")
        joblib.dump(lda_model, model_path); joblib.dump(vectorizer, vec_path)

    topics_words = extract_topics_top_words(lda_model, vocabulary, topn=10)
    diversity, coh_cv, coh_npmi = compute_topic_metrics(topics_words, tokenized_docs)

    doc_topic_prob = lda_model.transform(X_matrix)  # float64
    balance, effective_k, min_topic_size, max_topic_size = compute_topic_balance(doc_topic_prob)

    return dict(
        k=k_topics,
        learning_method=str(cfg['learning_method']),
        batch_size=int(cfg['batch_size']),
        max_iter=int(cfg['max_iter']),
        learning_decay=float(cfg['learning_decay']),
        coherence_cv=float(coh_cv),
        coherence_npmi=float(coh_npmi),
        diversity=float(diversity),
        balance=float(balance),
        effective_topics=int(effective_k),
        min_topic=int(min_topic_size),
        max_topic=int(max_topic_size),
        vocab_size=int(len(vocabulary)),
        n_docs=int(num_docs),
        model_path=model_path,
        vectorizer_path=vec_path,
        suffix=suffix
    )

# ============================ RUN GRID ==============================
with section("GRID SEARCH (LDA)"):
    p(f"grid_configurations lenght: {len(grid_configurations)}")
    batch_rows = Parallel(n_jobs=min(6, os.cpu_count()-1), backend='multiprocessing', verbose=5)(
        delayed(execute_single_configuration)(cfg) for cfg in grid_configurations
    )
    p("finito con batch_rows")

    if batch_rows:
        df_new = pd.DataFrame(batch_rows, columns=[c for c in result_columns if c != 'avg_score_robust'])

        # calcola avg_score_robust **in RAM**, non aggiunge altre colonne
        df_new['avg_score_robust'] = compute_avg_score_robust_in_memory(df_new)

        # merge (solo colonne ufficiali)
        if os.path.exists(out_path_grid_search_results):
            df_exist = pd.read_csv(out_path_grid_search_results)
            df_merged = pd.concat([df_exist[result_columns], df_new[result_columns]], ignore_index=True)
        else:
            df_merged = df_new[result_columns]

        df_merged = enforce_result_dtypes(df_merged)
        df_merged.to_csv(out_path_grid_search_results, index=False)
        p(f"#debug18 writing {len(df_new)} rows; total rows = {len(df_merged)}")

# ======================== COMPLETION FLAG ===========================
with open(os.path.join(base_root, "grid_search_lda_completed_successfully.txt"), "w") as f:
    f.write("LDA grid search completata con successo.\n")
p("#debug16 grid_search_lda_completed_successfully.txt written")
