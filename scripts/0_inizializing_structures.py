import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from bertopic import BERTopic
from hdbscan import HDBSCAN
from umap import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import torch
import pandas as pd
import time
from tqdm import tqdm
from glob import glob
import numpy as np
import sqlite3
from glob import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
import platform
from pathlib import Path
from bertopic import BERTopic
import pandas as pd
import plotly.io as pio



if platform.node().startswith("jupyter") or "cluster" in platform.node().lower():
    # on cluster
    extracted_dir = os.path.expanduser("~/telegram_2024/usc-tg-24-us-election/extracted")
else:
    # on laptop
    extracted_dir = os.path.join("..", "material", "extracted")

print("✅ Using path:", extracted_dir)


chats_path = '../material/chats.db'
conn = sqlite3.connect(chats_path)
cursor=conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables=cursor.fetchall()

# ========================
# 1. Leggi chats.db (SQLite)
# ========================

print("Tables in DB:", tables)
try:
    df_chats = pd.read_sql_query("SELECT * FROM chats", conn)
    df_chats =df_chats.drop_duplicates(subset='type_and_id')
    df_chats = df_chats.dropna(subset=['type_and_id'])
    print("number of unique chats", len(df_chats))
    print("chats.db - Tabella 'chats'")
    print(df_chats.head())
except Exception as e:
    print("Errore nel leggere la tabella:", e)

conn.close()

# ========================
# 2. Leggi discovery_edges.csv.gz
# ========================
try:
    df_edges = pd.read_csv('../material/discovery_edges.csv.gz')
    df_edges = df_edges.drop_duplicates(subset='type_and_id')
    df_edges = df_edges.dropna(subset=['type_and_id'])
    print("✅ discovery_edges.csv.gz, \n" \
    "Il timestamp da l'ultima volta che hanno visitato quel gruppo ma questo significa che non è davvero indicativo di una timeline \n")
    print(df_edges.head())
except Exception as e:
    print("Errore nel leggere discovery_edges:", e)

# ========================
# 3. Leggi first_nodes.csv.gz
# ========================
try:
    df_first_nodes = pd.read_csv('../material/first_nodes.csv.gz')
    print("number of non unique first nodes", len(df_first_nodes))
    df_first_nodes = df_first_nodes.drop_duplicates(subset='type_and_id')
    df_first_nodes = df_first_nodes.dropna(subset=['type_and_id'])
    print("✅ first_nodes.csv.gz")
    print(df_first_nodes.head())
    print("number of unique first nodes", len(df_first_nodes))
except Exception as e:
    print("Errore nel leggere first_nodes:", e)

seed_parents = (
    df_first_nodes['type_and_id']
    .dropna()
    .drop_duplicates()
)

df_children = (
    df_edges.loc[df_edges['parent'].isin(seed_parents), ['type_and_id']]
    .dropna(subset=['type_and_id'])
    .drop_duplicates()
    .reset_index(drop=True)
)

print(f"✅ children (level-1) ottenuti: {len(df_children)}")
print(df_children.head(10))

# (opzionale) salvo su disco
out_path = '../material/children_level1.csv.gz'
df_children.to_csv(out_path, index=False)
print(f"💾 salvato: {out_path}")



print("type_and_id unique in df_first_nodes" + str(df_first_nodes.type_and_id.nunique()))
print("type_and_id in df_first_nodes" + str(len(df_first_nodes)))
print("type_and_id NaN in df_first_nodes " + str(df_first_nodes['type_and_id'].isna().sum()))

