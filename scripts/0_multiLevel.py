#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from glob import glob
import pandas as pd
import sqlite3
import platform
from tqdm import tqdm
import argparse
import gzip
from joblib import Parallel, delayed
import gc

# ========================
# 0. Argomenti da riga di comando
# ========================

parser = argparse.ArgumentParser(description="Conta gruppi e messaggi per livello di crawling.")
parser.add_argument(
    "--max-level",
    type=int,
    default=10,
    help="Profondità massima di livelli da esplorare (default: 10)",
)
args = parser.parse_args()
MAX_LEVEL = args.max_level

print(f"🔧 MAX_LEVEL impostato a: {MAX_LEVEL}")

# ========================
# 1. Path degli estratti Telegram
# ========================

if platform.node().startswith("jupyter") or "cluster" in platform.node().lower():
    # cluster
    extracted_dir = os.path.expanduser("~/telegram_2024/usc-tg-24-us-election/extracted")
else:
    # laptop
    extracted_dir = os.path.join("..", "material", "extracted")

print("Using path:", extracted_dir)

# Root risultati (per coerenza con 99.py)
RESULTS_ROOT = os.path.join("..", "results", "levels")
os.makedirs(RESULTS_ROOT, exist_ok=True)

# ========================
# 2. Leggi discovery_edges e first_nodes
# ========================

df_edges = pd.read_csv("../material/discovery_edges.csv.gz")
# ci servono almeno parent e type_and_id
df_edges = df_edges.dropna(subset=["parent", "type_and_id"])
df_edges = df_edges.drop_duplicates(subset=["parent", "type_and_id"])

df_first_nodes = pd.read_csv("../material/first_nodes.csv.gz")
seed_parents = (
    df_first_nodes["type_and_id"]
    .dropna()
    .drop_duplicates()
)

print(f"🔹 seed parents (level 0): {len(seed_parents)}")

# ========================
# 3. Funzione: conta messaggi di un canale
# ========================

def count_messages_for_channel(channel_id: str, base_dir: str) -> int:
    """
    Conta il numero di messaggi per un canale leggendo tutti i file:
    <base_dir>/<channel_id>/YYYY-MM.tsv.gz
    """
    channel_path = os.path.join(base_dir, channel_id)
    if not os.path.isdir(channel_path):
        return 0

    # pattern tipo: 2024-01.tsv.gz, 2023-12.tsv.gz
    files = glob(os.path.join(channel_path, "[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz"))
    if not files:
        return 0

    total = 0
    for f in files:
        try:
            # leggo solo la colonna text per contare le righe = messaggi
            total += len(pd.read_csv(f, sep="\t", compression="gzip", usecols=["text"]))
        except Exception as e:
            print(f"⚠️ Errore leggendo {f} per {channel_id}: {e}")
    return total

# ========================
# 4. BFS sui livelli con conteggio gruppi/messaggi
# ========================

visited = set(seed_parents)           # canali già visti (evita loop)
current_level_nodes = list(seed_parents)

stats = []  # se poi vuoi usare i risultati in pandas

# puoi tarare n_jobs se vuoi limitare la RAM / I/O
N_JOBS = min(8, os.cpu_count() or 1)

for level in range(MAX_LEVEL):
    if not current_level_nodes:
        print(f"🔚 Nessun nodo al livello {level}, stop.")
        break

    # gruppi unici a questo livello
    df_nodes_level = pd.DataFrame({"type_and_id": sorted(set(current_level_nodes))}) 
    n_groups_for_current_level = len(df_nodes_level)
    del current_level_nodes 
    gc.collect()

    # ============================
    # 4a. Salva i nodi del livello
    # ============================

    # directory: ../results/levels/level_{level}/preProcessing/
    level_dir = os.path.join(RESULTS_ROOT, f"level_{level}", "preProcessing")
    os.makedirs(level_dir, exist_ok=True)

    # file: nodes_level_{level}.csv.gz con colonna type_and_id
    nodes_path = os.path.join(level_dir, f"nodes_level_{level}.csv.gz")
    df_nodes_level.to_csv(nodes_path, index=False, compression="gzip")

    print(f"Salvati {len(df_nodes_level)} nodi del livello {level} in: {nodes_path}")

    # ============================
    # 4b. Conta messaggi (parallel)
    # ============================

    print(f"\nLivello {level}: conteggio messaggi...")
    results = Parallel(n_jobs=N_JOBS)(
        delayed(count_messages_for_channel)(ch, extracted_dir)
        for ch in df_nodes_level['type_and_id']
    )

    total_msgs = sum(results)

    print(f"Level {level} -> unique groups for current level = {n_groups_for_current_level}, total messages = {total_msgs}")

    stats.append({
        "level": level,
        "n_groups": n_groups_for_current_level,
        "n_messages": total_msgs,
    })

    # ============================
    # 4c. Trova i figli per il livello successivo
    # ============================

    current_level_nodes = []
    for child in df_edges[df_edges["parent"].isin(df_nodes_level["type_and_id"])]["type_and_id"].dropna().unique():
        if child not in visited:
            visited.add(child)
            current_level_nodes.append(child)

    if not current_level_nodes:
        print(f"🔚 Nessun nuovo figlio dal livello {level}, stop BFS.")
        break

# opzionale: DataFrame riassuntivo in RAM
df_stats = pd.DataFrame(stats)
print("\nRiassunto livelli:")
print(df_stats)
df_stats.to_csv('levels_recap.csv', index=False)
