#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import random
import argparse
import shutil
import joblib
import numpy as np
import pandas as pd
import torch

from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from sklearn.feature_extraction.text import CountVectorizer

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from joblib import Parallel, delayed
from contextlib import contextmanager

# ===== LOGGER =================================================
START = time.perf_counter()
_last = [START]

def p(msg: str):
    now = time.perf_counter()
    tot = now - START
    delta = now - _last[0]
    print(f"[{tot:8.2f}s][+{delta:6.2f}s] {msg}")
    _last[0] = now

@contextmanager
def section(name: str):
    p(f"▶ START {name}")
    t0 = time.perf_counter()
    try:
        yield
    finally:
        dt = time.perf_counter() - t0
        p(f"■ END   {name} ({dt:.2f}s)")

# ===== SEED / DEVICE =========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "true"

device = "cuda" if torch.cuda.is_available() else "cpu"
p(f"device={device} cpu_count={os.cpu_count()}")

# ===== ARGPARSE ==============================================
parser = argparse.ArgumentParser(description="Grid search")
parser.add_argument("--input", type=str, default="0",
                    help="Livello di profondità")
parser.add_argument("--resume", action="store_true",
                    help="Riprende senza cancellare modelli/vectorizer/CSV")

args = parser.parse_args()
level_depth = args.input
p(f"#debug1 level_depth={level_depth}")

# ===== PATHS =================================================
input_path = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz"
output_path_df_sampled = f"../results/levels/level_{level_depth}/grid_search/df_sampled_level_{level_depth}.csv"
out_path_grid_search_results = f"../results/levels/level_{level_depth}/grid_search/grid_search_results_level_{level_depth}.csv"
dir_models = f"../results/levels/level_{level_depth}/grid_search/bertopic_models_level_{level_depth}/"
dir_vectorizers = f"../results/levels/level_{level_depth}/grid_search/vectorizers_level_{level_depth}/"
dir_embeddings = f"../results/levels/level_{level_depth}/grid_search/embeddings_level_{level_depth}/"
os.makedirs(dir_models, exist_ok=True)
os.makedirs(dir_vectorizers, exist_ok=True)
os.makedirs(dir_embeddings, exist_ok=True)
p(f"#debug2 dirs ready: {dir_models}, {dir_vectorizers}, {dir_embeddings}")

# ===== PULIZIA (condizionale) =================================
def empty_dir(dir_path: str):
    if os.path.isdir(dir_path):
        for name in os.listdir(dir_path):
            full = os.path.join(dir_path, name)
            if os.path.isfile(full) or os.path.islink(full):
                os.unlink(full)
            else:
                shutil.rmtree(full)

if not args.resume:
    empty_dir(dir_models)
    empty_dir(dir_vectorizers)
    p("#debug3 emptied models/vectorizers dirs")
    for f in [output_path_df_sampled, out_path_grid_search_results]:
        if os.path.exists(f):
            os.remove(f)
            p(f"[INFO] Rimosso file {f}")
    p("#debug4 removed previous CSVs if existed")
    p("[INFO] Pulizia completata: modelli, vectorizer e CSV resettati.")
else:
    p("[RESUME] Nessuna pulizia: riuso modelli/vectorizer/CSV esistenti.")

# ===== LETTURA DATI ==========================================
with section("READ TSV"):
    df_pre = pd.read_csv(input_path, sep='\t', compression='gzip')
    p("input accepted head:\n" + str(df_pre.head()))
    p(f"len :{len(df_pre)}")

df_sampled = df_pre.sample(frac=1, random_state=SEED)
df_sampled.to_csv(output_path_df_sampled, index=False)
p(f"#debug6 df_sampled len={len(df_sampled)} saving to {output_path_df_sampled}")
p("imported df_sampled")

texts = [str(s).split() for s in df_sampled['text_preprocessed'].tolist()]

# ===== METRICHE ==============================================
def get_metrics(topic_model, texts=texts):
    topics = topic_model.get_topics()
    topics_list = []
    for tid, words in topics.items():
        if tid == -1:
            continue
        ws = [w for w, _ in words if w]
        if len(ws) >= 10:
            topics_list.append(ws)
    if not topics_list:
        return 0.0, 0.0
    diversity = TopicDiversity(topk=10).score({"topics": topics_list})
    coherence = Coherence(topk=10, texts=texts).score({"topics": topics_list})
    return diversity, coherence

# ===== SINGLE RUN ============================================
def run_single_run(model_name, embeddings, umap_config, hdbscan_config):
    p(f"#run start model={model_name} umap={umap_config} hdbscan={hdbscan_config}")
    t_run = time.perf_counter()

    umap_model = UMAP(**umap_config, random_state=SEED)
    hdbscan_model = HDBSCAN(**hdbscan_config)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
        top_n_words=20,
        language='english'
    )

    t0 = time.perf_counter()
    topic_model.fit_transform(df_sampled['text_preprocessed'], embeddings=embeddings)
    p(f"fit_transform: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    diversity, coherence = get_metrics(topic_model)
    p(f"metrics: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    reduced = umap_model.transform(embeddings)
    labels = np.array(topic_model.topics_)
    mask = labels != -1
    sil = silhouette_score(reduced[mask], labels[mask]) if mask.sum() > 1 and len(np.unique(labels[mask])) > 1 else 0.0
    p(f"silhouette: {time.perf_counter()-t0:.2f}s")

    t0 = time.perf_counter()
    suffix = f"{model_name}_umap{umap_config['n_components']}_umap{umap_config['n_neighbors']}_umap{umap_config['min_dist']}_hdbscan{hdbscan_config['min_cluster_size']}"
    topic_model.save(os.path.join(dir_models, suffix))
    joblib.dump(topic_model.vectorizer_model, os.path.join(dir_vectorizers, f"vectorizer_{suffix}.pkl"))
    p(f"save: {time.perf_counter()-t0:.2f}s")

    series = pd.Series(labels[labels != -1])
    result = {
        'model': model_name,
        'umap_n_components': umap_config['n_components'],
        'umap_n_neighbors': umap_config['n_neighbors'],
        'umap_min_dist': umap_config['min_dist'],
        'hdbscan_min_cluster_size': hdbscan_config['min_cluster_size'],
        'coherence': coherence,
        'diversity': diversity,
        'silhouette': sil,
        'n_outliers': int((labels == -1).sum()),
        'n_topics': int(len(np.unique(labels)) - (-1 in labels)),
        'min_topic': int(series.value_counts().min()) if not series.empty else 0,
        'max_topic': int(series.value_counts().max()) if not series.empty else 0
    }
    p(f"#run end topics={result['n_topics']} outliers={result['n_outliers']} sil={sil:.4f} total={time.perf_counter()-t_run:.2f}s")
    return result

# ===== VECTORIZE (placeholder, come nel tuo) =================
vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")

# ===== GRIGLIA COMPLETA (come la tua) ========================
umap_params = [
    {'n_components': 10, 'n_neighbors': 5,   'min_dist': 0.0},
    {'n_components': 10, 'n_neighbors': 25,  'min_dist': 0.0},
    {'n_components': 10, 'n_neighbors': 5,   'min_dist': 0.1},
    {'n_components': 10, 'n_neighbors': 25,  'min_dist': 0.1},
    {'n_components': 5,  'n_neighbors': 5,   'min_dist': 0.0},
    {'n_components': 5,  'n_neighbors': 25,  'min_dist': 0.0},
    {'n_components': 5,  'n_neighbors': 5,   'min_dist': 0.1},
    {'n_components': 5,  'n_neighbors': 25,  'min_dist': 0.1},
    {'n_components': 3,  'n_neighbors': 5,   'min_dist': 0.0},
    {'n_components': 3,  'n_neighbors': 25,  'min_dist': 0.0},
    {'n_components': 3,  'n_neighbors': 5,   'min_dist': 0.1},
    {'n_components': 3,  'n_neighbors': 25,  'min_dist': 0.1},
]

hdbscan_params = [
    {'min_cluster_size': 10},
    {'min_cluster_size': 15},
    {'min_cluster_size': 30},
    {'min_cluster_size': 50},
    {'min_cluster_size': 90},
]

# ===== CSV RISULTATI (come il tuo) ===========================
cols = [
    'model','umap_n_components','umap_n_neighbors','umap_min_dist',
    'hdbscan_min_cluster_size','coherence','diversity','silhouette',
    'n_outliers','n_topics','min_topic','max_topic'
]
if not os.path.exists(out_path_grid_search_results):
    pd.DataFrame(columns=cols).to_csv(out_path_grid_search_results, index=False)
    p(f"#debug8 created results file: {out_path_grid_search_results}")
else:
    p(f"#debug8 results file exists: {out_path_grid_search_results}")

# ===== MODELLI SBERT (come il tuo) ===========================
models = {
    'all-distilroberta-v1': SentenceTransformer('all-distilroberta-v1'),
    'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2')
}
p(f"#debug9 models ready: {list(models.keys())}")

# ===== BUILD EMBEDDINGS (skip se già esistono) ===============
with section("BUILD EMBEDDINGS"):
    for model_name, model_instance in models.items():
        path_emb = f"{dir_embeddings}/embedding_{model_name}_level_{level_depth}.npy"
        if os.path.exists(path_emb):
            p(f"Embedding già esistente: {model_name}")
            continue
        p(f"#debug10 building embeddings for {model_name}")
        model_instance = model_instance.to(device)
        t0 = time.perf_counter()
        embeddings = model_instance.encode(
            df_sampled['text_preprocessed'].tolist(),
            show_progress_bar=True,
            device=device
        )
        np.save(path_emb, embeddings)
        p(f"Salvato: {path_emb}")
        p(f"#debug11 embeddings done for {model_name} in {time.perf_counter()-t0:.2f}s shape={embeddings.shape}")

# ===== HELPER: SKIP CONFIG GIÀ TESTATE =======================
def not_already_tested(model_name, uc, hc):
    df = pd.read_csv(out_path_grid_search_results)
    return not ((df['model'] == model_name) &
                (df['umap_n_components'] == uc['n_components']) &
                (df['umap_n_neighbors'] == uc['n_neighbors']) &
                (df['umap_min_dist'] == uc['min_dist']) &
                (df['hdbscan_min_cluster_size'] == hc['min_cluster_size'])).any()

# ===== GRID SEARCH PARALLELA ================================
with section("GRID SEARCH"):
    for model_name, model_instance in models.items():
        path_emb = f"{dir_embeddings}/embedding_{model_name}_level_{level_depth}.npy"
        if not os.path.exists(path_emb):
            p(f"[⚠️] Missing embedding for {model_name}, skipped.")
            continue

        embeddings = np.load(path_emb)
        p(f"#debug12 loaded embeddings {model_name} shape={embeddings.shape}")

        param_grid = [
            (uc, hc)
            for uc in umap_params
            for hc in hdbscan_params
            if not_already_tested(model_name, uc, hc)
        ]
        p(f"#debug13 {model_name} param_grid size={len(param_grid)}")

        p(f"#debug14 launching Parallel n_jobs={os.cpu_count()-1}")
        batch_results = Parallel(n_jobs=os.cpu_count() - 1, verbose=5)(
            delayed(run_single_run)(model_name, embeddings, uc, hc)
            for uc, hc in param_grid
        )

        temp_df = pd.DataFrame(batch_results)
        p(f"#debug18 writing {len(temp_df)} rows to results")
        temp_df.to_csv(out_path_grid_search_results, mode='a', index=False, header=False)

# ===== COMPLETION FLAG ======================================
with open("completed_successfully.txt", "w") as f:
    f.write("Grid search completata con successo.\n")
p("#debug16 completed_successfully.txt written")
