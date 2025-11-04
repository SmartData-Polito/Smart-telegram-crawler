#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BO (CPU-only) di BERTopic con KPCA(RBF) + HDBSCAN — MORE_TOPICS (FIX DTYPE + COSINE)

Cosa risolve:
- Errore "Buffer dtype mismatch, expected 'double_t' but got 'float'":
  * HDBSCAN in Cython richiede float64 → forziamo embeddings ridotti a **float64**.
- Errore con metric='cosine':
  * Forziamo **algorithm='generic'** quando metric='cosine' (evita BallTree/KDTree che non
    supportano cosine nella tua build).
- Aumenta il n. di topic:
  * cluster_selection_method='leaf', min_samples proporzionale a mcs, mcs ∈ [5,40] (tunable).
- KPI: silhouette calcolata con la **stessa metrica** del clustering.
- KPCA one-shot (fit/transform una volta sola fuori dalla BO).
"""


"""
python 4_grid_search_BO_algo.py \
  --input 0 \
  --embed_model all-MiniLM-L6-v2 \
  --bo_calls 12 --bo_random_starts 6 \
  --fp32 1 \
  --kpca_fit_subset 20000 \
  --kpca_var_threshold 0.80 \
  --kpca_components_cap 25 \
  --kpca_min_components 10 \
  --hdbscan_metric euclidean \
  --unitnorm_if_euclidean 1 \
  --hdbscan_selection leaf \
  --min_samples_ratio 0.10 \
  --mcs_lo 5 --mcs_hi 40 \
  --trace_cluster 1 --trace_topics 0

"""

import os, sys, time, random, argparse, warnings
from time import perf_counter
from contextlib import contextmanager

import numpy as np
import pandas as pd
import torch, psutil

from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler, normalize, FunctionTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence

from skopt import gp_minimize
from skopt.space import Integer
from skopt.utils import use_named_args

from hdbscan import HDBSCAN

warnings.filterwarnings("ignore")

# =============================
# Pretty logger
# =============================
START_TS = time.perf_counter(); _last = [START_TS]

def log(msg: str) -> None:
    now = time.perf_counter(); tot = now - START_TS; delta = now - _last[0]
    print(f"[{tot:8.2f}s][+{delta:6.2f}s] {msg}"); _last[0] = now

@contextmanager
def section(name: str):
    log(f"▶ START {name}"); t0 = perf_counter()
    try:
        yield
    finally:
        log(f"■ END   {name} ({perf_counter()-t0:.2f}s)")

# =============================
# CLI
# =============================
parser = argparse.ArgumentParser(description="CPU-only BO of BERTopic with KPCA(RBF)+HDBSCAN — MORE_TOPICS (fix)")
parser.add_argument("--input", type=str, default="0")

# Embeddings
parser.add_argument("--embed_model", type=str, default="all-MiniLM-L6-v2")
parser.add_argument("--fp32", type=int, default=1, choices=[0,1])

# CPU
parser.add_argument("--cpu_threads", type=int, default=max(1, os.cpu_count()-1))

# BO
parser.add_argument("--bo_calls", type=int, default=12)
parser.add_argument("--bo_random_starts", type=int, default=6)

# KPCA (RBF)
parser.add_argument("--kpca_fit_subset", type=int, default=20000)
parser.add_argument("--kpca_var_threshold", type=float, default=0.80)
parser.add_argument("--kpca_components_cap", type=int, default=25)
parser.add_argument("--kpca_min_components", type=int, default=10)

# HDBSCAN knobs
parser.add_argument("--hdbscan_metric", type=str, default="cosine", choices=["cosine","euclidean"])
parser.add_argument("--hdbscan_selection", type=str, default="leaf", choices=["leaf","eom"])
parser.add_argument("--min_samples_ratio", type=float, default=0.10)  # min_samples = round(ratio*mcs)
parser.add_argument("--mcs_lo", type=int, default=5)
parser.add_argument("--mcs_hi", type=int, default=40)
parser.add_argument("--unitnorm_if_euclidean", type=int, default=1, choices=[0,1])

# Tracing
parser.add_argument("--trace_cluster", type=int, default=1, choices=[0,1])
parser.add_argument("--trace_topics",  type=int, default=0, choices=[0,1])
CFG = parser.parse_args()

# =============================
# Seed & env
# =============================
SEED = 42
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"
for var in ["OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","VECLIB_MAXIMUM_THREADS","NUMEXPR_NUM_THREADS"]:
    os.environ[var] = str(CFG.cpu_threads)
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
try: torch.set_num_threads(CFG.cpu_threads)
except Exception: pass
log(f"[DATA] device=cpu  threads={CFG.cpu_threads}")

# =============================
# Paths
# =============================
level_id = CFG.input
level_dir = f"../results/levels/level_{level_id}"
path_in_tsv = f"{level_dir}/preProcessing/preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_id}.tsv.gz"
emb_dir    = f"{level_dir}/runtime_grid_search/embeddings_level_{level_id}/"
models_dir = f"{level_dir}/runtime_grid_search/bertopic_models_level_{level_id}/"
res_csv    = f"{level_dir}/runtime_grid_search/bo_results_level_{level_id}_KernelPCA_HDBSCAN_CPU_MORE_TOPICS_FIX.csv"
profile_csv= f"{level_dir}/runtime_grid_search/bo_profile_level_{level_id}_HDBSCAN_MORE_TOPICS_FIX.csv"
for d in (emb_dir, models_dir, level_dir, os.path.dirname(res_csv)):
    os.makedirs(d, exist_ok=True)

if not os.path.exists(profile_csv):
    with open(profile_csv, "w") as f:
        f.write(
            "try_idx,min_cluster_size,used_n_comp,"
            "t_kpca_fit,t_kpca_transform,"
            "t_hdbscan,t_topic_build,t_metrics,"
            "coh,div,sil,n_topics,"
            "rss_mb,vm_mb,total_s\n"
        )

_try_idx = {"i": 0}

# =============================
# Read data
# =============================
with section("READ TSV"):
    df = pd.read_csv(path_in_tsv, sep='\t', compression='gzip')
    log(f"[TIMING][READ TSV] rows={len(df):,}")
    df = df.sample(frac=1, random_state=SEED).copy()
    if int(len(df)/5) < CFG.kpca_fit_subset:
        CFG.kpca_fit_subset = int(len(df)/5)
        log(f"[DATA] kpca_fit_subset ridotto a {CFG.kpca_fit_subset} (≈ 1/6 delle righe)")

TEXTS = df['text_preprocessed'].astype(str).tolist()
TOKENIZED_TEXTS = [s.split() for s in TEXTS]

# =============================
# Embeddings
# =============================
emb_path = f"{emb_dir}/embedding_{CFG.embed_model}_level_{level_id}_fp{CFG.fp32}.npy"
with section("EMBEDDINGS"):
    if os.path.exists(emb_path):
        EMB = np.load(emb_path, mmap_mode=None)
        log(f"[IO] loaded {EMB.shape} dtype={EMB.dtype} from {emb_path} (~{os.path.getsize(emb_path)/1024/1024:.1f}MB)")
    else:
        st = SentenceTransformer(CFG.embed_model).to("cpu")
        EMB = st.encode(TEXTS, show_progress_bar=True, device="cpu", batch_size=512)
        if CFG.fp32:
            EMB = EMB.astype(np.float32, copy=False)
        np.save(emb_path, EMB)
        log(f"[IO] saved embeddings to {emb_path} (~{os.path.getsize(emb_path)/1024/1024:.1f}MB)")

VECT = CountVectorizer(ngram_range=(1,1), stop_words="english")

# =============================
# KPCA helpers
# =============================

def explain_gamma(gamma: float) -> str:
    sigma = float(np.sqrt(1.0/(2.0*gamma)))
    return f"gamma={gamma:.4g} (RBF) ⇒ σ≈{sigma:.3f}"

def median_heuristic_gamma(X: np.ndarray, subsample: int = 2000, pair_samples: int = 100_000):
    n = X.shape[0]
    if n < 2:
        return 1.0, None
    S = min(n, subsample)
    rng = np.random.default_rng(SEED)
    idx = rng.choice(n, size=S, replace=False)
    Xs = X[idx].astype(np.float32, copy=False)
    max_pairs = S * (S - 1) // 2
    M = int(min(pair_samples, max_pairs))
    i = rng.integers(0, S, size=M); j = rng.integers(0, S, size=M)
    mask = i != j; i = i[mask]; j = j[mask]
    if i.size == 0:
        return 1.0, None
    diffs = Xs[i] - Xs[j]
    d2 = np.einsum('ij,ij->i', diffs, diffs, optimize=True)
    r_med = float(np.sqrt(np.median(d2)))
    if not np.isfinite(r_med) or r_med <= 0:
        return 1.0, None
    gamma0 = 1.0 / (2.0 * (r_med ** 2) + 1e-12)
    return float(gamma0), r_med

# =============================
# KPCA one-shot
# =============================
GAMMA_FIXED, r_med = median_heuristic_gamma(EMB, subsample=2000, pair_samples=100_000)
log(f"[KPCA] Using fixed {explain_gamma(GAMMA_FIXED)}" + (f" | r_med≈{r_med:.3f}" if r_med else " | heuristic fallback"))

with section("KPCA-FIT+TRANSFORM (ONE-SHOT)"):
    kpca = KernelPCA(n_components=CFG.kpca_components_cap, kernel="rbf", gamma=GAMMA_FIXED, random_state=SEED)
    if CFG.kpca_fit_subset and len(EMB) > CFG.kpca_fit_subset:
        rng = np.random.default_rng(SEED)
        idx = rng.choice(len(EMB), size=CFG.kpca_fit_subset, replace=False)
        kpca.fit(EMB[idx])
        log(f"[TIMING][KPCA-FIT] subset={CFG.kpca_fit_subset:,}, cap={CFG.kpca_components_cap}, {explain_gamma(GAMMA_FIXED)}")
    else:
        kpca.fit(EMB)
        log(f"[TIMING][KPCA-FIT] subset=ALL({len(EMB):,}), cap={CFG.kpca_components_cap}, {explain_gamma(GAMMA_FIXED)}")

    eigvals = np.asarray(getattr(kpca, "eigenvalues_", getattr(kpca, "lambdas_", [])), dtype=np.float64)
    total = float(np.sum(eigvals)) if eigvals.size else 0.0
    if total > 0:
        cum = np.cumsum(eigvals) / total
        used_n = int(np.searchsorted(cum, CFG.kpca_var_threshold) + 1)
        used_n = max(CFG.kpca_min_components, min(used_n, eigvals.size, CFG.kpca_components_cap))
    else:
        used_n = CFG.kpca_min_components

    RED = kpca.transform(EMB)[:, :used_n]
    # StandardScaler → float64; poi garantiamo **float64** per HDBSCAN
    RED_STD = StandardScaler(with_mean=True, with_std=True).fit_transform(RED).astype(np.float64, copy=False)

    # Se euclidean e vogliamo approssimare cosine → L2-norm (manteniamo float64)
    if CFG.hdbscan_metric == 'euclidean' and CFG.unitnorm_if_euclidean:
        RED_STD = normalize(RED_STD, norm='l2', copy=False)

    log(f"[KPCA] used_n={used_n} | RED_STD shape={RED_STD.shape} dtype={RED_STD.dtype}")
    _RED_STD = RED_STD; _USED_N = used_n

# =============================
# BO space
# =============================
TARGET_CLUSTERS_LO = 50
TARGET_CLUSTERS_HI = 150
mcs_space = Integer(CFG.mcs_lo, CFG.mcs_hi, name="min_cluster_size")
space = [mcs_space]

best = {"score": -1.0, "params": None, "metrics": None, "model": None}

# =============================
# Utils
# =============================

def time_cluster_only(clusterer, X):
    t0 = perf_counter(); labels = clusterer.fit_predict(X); dt = perf_counter() - t0
    return labels, float(dt)

def compute_avg_score(coh: float, div: float, sil: float):
    coh_p = (coh + 1.0) / 2.0
    sil_p = (sil + 1.0) / 2.0
    div_p = float(min(max(div, 0.0), 1.0))
    return float((coh_p + div_p + sil_p) / 3.0), (coh_p, div_p, sil_p)

def cluster_count_penalty(k: int, lo: int = TARGET_CLUSTERS_LO, hi: int = TARGET_CLUSTERS_HI) -> float:
    if k <= 5: return 0.3
    if lo <= k <= hi: return 1.0
    dist = (lo - k)/lo if k < lo else (k - hi)/hi
    alpha = 2.0
    return 1.0 / (1.0 + alpha * abs(dist))


def safe_silhouette(X: np.ndarray, labels: np.ndarray, metric: str) -> float:
    mask = labels != -1
    if mask.sum() < 2: return 0.0
    uniq = np.unique(labels[mask])
    if uniq.size <= 1: return 0.0
    try:
        return float(silhouette_score(X[mask], labels[mask], metric=metric))
    except Exception:
        return 0.0

# =============================
# Score configuration
# =============================

def score_configuration(min_cluster_size: int) -> dict:
    t_all0 = perf_counter()
    RED_STD = _RED_STD; used_n = _USED_N

    ms = max(1, int(round(CFG.min_samples_ratio * min_cluster_size)))
    algo = 'generic' if CFG.hdbscan_metric == 'cosine' else 'best'

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=ms,
        metric=CFG.hdbscan_metric,
        algorithm=algo,
        cluster_selection_method=CFG.hdbscan_selection,
        cluster_selection_epsilon=0.0,
        core_dist_n_jobs=CFG.cpu_threads,
    )

    real_time = None
    if CFG.trace_cluster:
        labels_tmp, real_time = time_cluster_only(clusterer.__class__(**clusterer.get_params()), RED_STD)
        uniq = int(np.unique(labels_tmp[labels_tmp!=-1]).size)
        log(f"[TIMING][HDBSCAN] mcs={min_cluster_size} ms={ms} metric={CFG.hdbscan_metric}/{algo} → {real_time:.2f}s (uniq clusters={uniq}) [standalone]")

    identity = FunctionTransformer(validate=False)
    topic_model = BERTopic(
        embedding_model=None,
        umap_model=identity,
        hdbscan_model=clusterer,
        top_n_words=10,
        language='english',
        vectorizer_model=VECT,
        calculate_probabilities=False,
        verbose=False,
    )

    t_bt0 = perf_counter()
    topics, _ = topic_model.fit_transform(TEXTS, embeddings=RED_STD)
    t_block = perf_counter() - t_bt0

    if real_time is not None and real_time <= t_block:
        t_hdb = real_time; t_topic_build = t_block - real_time
    else:
        t_hdb = min(t_block * 0.60, t_block); t_topic_build = t_block - t_hdb
    log(f"[TIMING][BERTopic] fit_transform: {t_block:.2f}s | split(hdbscan={t_hdb:.2f}s, topics≈{t_topic_build:.2f}s)")

    labels = np.array(topics)

    # Metriche
    t_m0 = perf_counter()
    topics_dict = topic_model.get_topics(); topic_wordlists = []
    for tid, words in topics_dict.items():
        if tid == -1: continue
        topw = [w for w, _ in words if w]
        if len(topw) >= 10: topic_wordlists.append(topw)

    if not topic_wordlists:
        diversity = coherence = silhouette = 0.0; n_topics = 0
    else:
        diversity  = TopicDiversity(topk=10).score({"topics": topic_wordlists})
        coherence = Coherence(topk=10, measure="c_npmi", texts=TOKENIZED_TEXTS).score({"topics": topic_wordlists})
        silhouette = safe_silhouette(RED_STD, labels, metric=CFG.hdbscan_metric)
        lab_valid = labels[labels != -1]
        n_topics = int(np.unique(lab_valid).size) if lab_valid.size else 0
    t_metrics = perf_counter() - t_m0
    log(f"[TIMING][METRICS] coh={coherence:.4f} div={diversity:.4f} sil={silhouette:.4f} for n_topics={n_topics} in {t_metrics:.2f}s")

    pen = cluster_count_penalty(n_topics)
    total_s = perf_counter() - t_all0
    rss = psutil.Process(os.getpid()).memory_info().rss/1024/1024
    vms = psutil.Process(os.getpid()).memory_info().vms/1024/1024
    log(f"[MEM] rss={rss:,.1f}MB vms={vms:,.1f}MB | total_try={total_s:.2f}s | penalty={pen:.3f}")

    with open(profile_csv, "a") as f:
        f.write(
            f"{_try_idx['i']},{min_cluster_size},{used_n},{0.0:.3f},{0.0:.3f},{t_hdb:.3f},{t_topic_build:.3f},{t_metrics:.3f},{coherence:.4f},{diversity:.4f},{silhouette:.4f},{n_topics},{rss:.1f},{vms:.1f},{total_s:.3f}\n"
        )

    return dict(coherence=coherence, diversity=diversity, silhouette=silhouette,
                n_topics=n_topics, model=topic_model, used_n_comp=used_n,
                total_s=total_s, penalty=pen, min_cluster_size=min_cluster_size)

@use_named_args(space)
def objective(**params):
    _try_idx["i"] += 1
    mcs = int(params["min_cluster_size"])
    res = score_configuration(mcs)

    avg_score, _ = compute_avg_score(res["coherence"], res["diversity"], res["silhouette"])
    final_score = avg_score * res["penalty"]

    if final_score > best["score"]:
        best["score"] = final_score
        best["params"] = dict(n_components=res["used_n_comp"], kpca_gamma=GAMMA_FIXED,
                               min_cluster_size=mcs, metric=CFG.hdbscan_metric,
                               min_samples_ratio=CFG.min_samples_ratio, selection=CFG.hdbscan_selection)
        best["metrics"] = dict(coherence=res["coherence"], diversity=res["diversity"],
                                silhouette=res["silhouette"], n_topics=res["n_topics"], seconds=res["total_s"],
                                avg_score=avg_score, penalty=res["penalty"], final_score=final_score)
        best["model"] = res["model"]

    log(
        f"[TRY #{_try_idx['i']:02d}] comps={res['used_n_comp']:2d}  mcs={mcs:3d}  ->  k={res['n_topics']:3d}  "
        f"coh={res['coherence']:.4f}  div={res['diversity']:.4f}  sil={res['silhouette']:.4f}  "
        f"AVG={avg_score:.4f}  PEN={res['penalty']:.3f}  FINAL={final_score:.4f}  time={res['total_s']:.1f}s\n"
    )
    return -final_score

# =============================
# RUN BO
# =============================
with section("BAYESIAN OPTIMIZATION (scikit-optimize, CPU)"):
    try:
        _ = gp_minimize(func=objective, dimensions=space, n_calls=CFG.bo_calls,
                        n_random_starts=CFG.bo_random_starts, acq_func="EI", random_state=SEED)
    except KeyboardInterrupt:
        log("[WARN] BO interrotta da tastiera; userò il best corrente se presente.")
    except Exception as e:
        log(f"[ERROR] BO failed: {e}")
        if best["params"] is None:
            sys.exit(1)

# =============================
# Save best
# =============================
if best["params"] is None or best["model"] is None:
    log("[ERROR] Nessun best valido trovato; skip salvataggio.")
    sys.exit(1)

log(f"[BEST] params={best['params']}  metrics={best['metrics']}")

best_name = (
    f"CPU_BO_KPCA_rbf_nc{best['params']['n_components']}"
    f"_g{best['params']['kpca_gamma']:.4g}__HDBSCAN_mcs{best['params']['min_cluster_size']}"
    f"_metric{best['params']['metric']}_msr{best['params']['min_samples_ratio']:.2f}_sel{best['params']['selection']}"
    f"__{CFG.embed_model}_fp{CFG.fp32}"
)

best_path = os.path.join(models_dir, best_name)
best["model"].save(best_path)
log(f"[SAVED] model -> {best_path}")

row = pd.DataFrame([{
    "embed_model": CFG.embed_model,
    "kpca_n_components": best['params']['n_components'],
    "kpca_kernel": "rbf",
    "kpca_gamma": best['params']['kpca_gamma'],
    "hdbscan_min_cluster_size": best['params']['min_cluster_size'],
    "hdbscan_metric": best['params']['metric'],
    "hdbscan_min_samples_ratio": best['params']['min_samples_ratio'],
    "hdbscan_selection": best['params']['selection'],
    **best["metrics"],
}])
row.to_csv(res_csv, mode='a', index=False, header=(not os.path.exists(res_csv)), sep=';')
log(f"[SAVED] CSV -> {res_csv} (sep=';')")

log("Run completed. Suggerimenti:")
log("- Se ancora pochi topic: prova --min_samples_ratio 0.05 o --mcs_lo 3 --mcs_hi 30")
log("- Se troppi micro-cluster: alza --min_samples_ratio 0.20 o --hdbscan_selection eom")
