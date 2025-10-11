#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
python dump_topics_to_text.py \
  --input 0 \
  --suffix paraphrase-MiniLM-L6-v2_umap5_umap5_umap0.0_hdbscan30 \
  --label coherence \
  --out topics_coherence.txt \
  --top_n 20

'''

import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
from bertopic import BERTopic

'''
python 2_topics_words.py   --input 0   --suffix paraphrase-MiniLM-L6-v2_uma
p5_umap5_umap0.0_hdbscan30   --label avg_score   --out topics_L0_avg_forced.txt   --top_
n 20

python 2_topics_words.py --input 1 --suffix paraphrase-MiniLM-L6-v2_umap10_umap5_umap0.0_hdbscan30 --label coherence --out topics_L1_avg_forced.txt --top_n 20
'''

# --------------------
# Argomenti
# --------------------
parser = argparse.ArgumentParser(description="Esporta i topic BERTopic in un file di testo (topicX: parole).")
parser.add_argument("--input", type=str, default="0", help="level depth (e.g., 0, 1, 2, ...)")
parser.add_argument("--choice", type=str, default="coherence",
                    choices=["silhouette", "coherence", "diversity", "avg_score"],
                    help="criterio per selezionare il best model dal CSV (usato se --suffix non è passato)")
parser.add_argument("--suffix", type=str, default=None,
                    help=("Suffix esatto del modello da caricare, es. "
                          "'paraphrase-MiniLM-L6-v2_umap5_umap5_umap0.0_hdbscan30'. "
                          "Se presente, ignora --choice."))
parser.add_argument("--label", type=str, default="coherence",
                    help="Prima riga del file (es. 'coherence', 'diversity', 'silhouette', ecc.)")
parser.add_argument("--out", type=str, default="topics.txt",
                    help="Nome file di output (testuale) nella cartella corrente.")
parser.add_argument("--top_n", type=int, default=20,
                    help="Numero di parole per topic.")
parser.add_argument("--include-outlier", action="store_true",
                    help="Se passato, include anche il topic -1 (outlier).")
args = parser.parse_args()

level_depth = args.input
choice = args.choice
suffix_arg = args.suffix.strip() if args.suffix else None
if suffix_arg and (suffix_arg.startswith("/") or suffix_arg.startswith("\\")):
    suffix_arg = suffix_arg[1:]

# --------------------
# Paths base progetto
# --------------------
base_dir = f"../../results/levels/level_{level_depth}"
grid_dir = os.path.join(base_dir, "grid_search")
bertopic_models_dir = os.path.join(grid_dir, f"bertopic_models_level_{level_depth}")
embeddings_dir = os.path.join(grid_dir, f"embeddings_level_{level_depth}")
grid_search_results_path = os.path.join(grid_dir, f"grid_search_results_level_{level_depth}.csv")
df_sampled_path = os.path.join(grid_dir, f"df_sampled_level_{level_depth}.csv")

# --------------------
# Pre-check minimi
# --------------------
missing = []
for pth in [grid_search_results_path, bertopic_models_dir]:
    if not os.path.exists(pth):
        missing.append(pth)
if missing and not suffix_arg:
    print("[ERR] Mancano file/cartelle necessari e non hai passato --suffix:")
    for m in missing:
        print("  -", m)
    raise SystemExit(1)

# --------------------
# Utility: parse suffix
# --------------------
_suffix_re = re.compile(
    r'^(?P<model>.+?)_umap(?P<ncomp>[^_]+)_umap(?P<nneigh>[^_]+)_umap(?P<mindist>[^_]+)_hdbscan(?P<mcs>\d+)$'
)

def parse_suffix(s: str):
    m = _suffix_re.match(s)
    if not m:
        raise ValueError(f"Suffix non nel formato atteso: {s}")
    gd = m.groupdict()
    return {
        "model": gd["model"],
        "umap_n_components": int(float(gd["ncomp"])),
        "umap_n_neighbors": int(float(gd["nneigh"])),
        "umap_min_dist": float(gd["mindist"]),
        "hdbscan_min_cluster_size": int(gd["mcs"]),
    }

# --------------------
# Determina il modello da caricare (suffix o scelta da CSV)
# --------------------
if suffix_arg:
    parsed = parse_suffix(suffix_arg)
    best_model_name = parsed["model"]
    uc = {
        "n_components": parsed["umap_n_components"],
        "n_neighbors": parsed["umap_n_neighbors"],
        "min_dist": parsed["umap_min_dist"],
    }
    hc = {"min_cluster_size": parsed["hdbscan_min_cluster_size"]}
    suffix = (
        f"{best_model_name}_"
        f"umap{uc['n_components']}_"
        f"umap{uc['n_neighbors']}_"
        f"umap{uc['min_dist']}_"
        f"hdbscan{hc['min_cluster_size']}"
    )
else:
    # Leggi CSV e seleziona per metrica
    df_grid = pd.read_csv(grid_search_results_path)
    if "avg_score" not in df_grid.columns:
        df_grid["avg_score"] = (df_grid["silhouette"] + df_grid["coherence"] + df_grid["diversity"]) / 3

    required_cols = {
        "model", "umap_n_components", "umap_n_neighbors", "umap_min_dist",
        "hdbscan_min_cluster_size", "silhouette", "coherence", "diversity", "avg_score"
    }
    if not required_cols.issubset(df_grid.columns):
        raise ValueError(f"CSV incompleto. Mancano colonne: {required_cols - set(df_grid.columns)}")

    best_row = df_grid.sort_values(by=choice, ascending=False).iloc[0]
    best_model_name = best_row["model"]
    uc = {
        "n_components": int(best_row["umap_n_components"]),
        "n_neighbors": int(best_row["umap_n_neighbors"]),
        "min_dist": float(best_row["umap_min_dist"]),
    }
    hc = {"min_cluster_size": int(best_row["hdbscan_min_cluster_size"])}
    suffix = (
        f"{best_model_name}_"
        f"umap{uc['n_components']}_"
        f"umap{uc['n_neighbors']}_"
        f"umap{uc['min_dist']}_"
        f"hdbscan{hc['min_cluster_size']}"
    )

# --------------------
# Carica il modello
# --------------------
model_path = Path(os.path.join(bertopic_models_dir, suffix))
if not model_path.exists():
    print(f"[ERR] Modello non trovato: {model_path}")
    raise SystemExit(1)

print(f"[INFO] Carico modello: {model_path.name}")
topic_model = BERTopic.load(str(model_path))

# --------------------
# Estrai topic e scrivi file
# --------------------
topics_info = topic_model.get_topic_info()
topic_ids = []
for t in topics_info["Topic"]:
    ti = int(t)
    if ti == -1 and not args.include_outlier:
        continue
    topic_ids.append(ti)
topic_ids = sorted(topic_ids)

lines = [args.label]
for t in topic_ids:
    words = [w for (w, _) in topic_model.get_topic(t)[:args.top_n]]
    lines.append(f"topic{t}: {', '.join(words)}")

out_path = Path(args.out).resolve()
out_path.write_text("\n".join(lines), encoding="utf-8")

print(f"[OK] Scritto: {out_path}  |  topic: {len(topic_ids)}  |  top_n={args.top_n}")
print(f"[OK] Etichetta prima riga: '{args.label}'")
print(f"[OK] Modello: {suffix}")
