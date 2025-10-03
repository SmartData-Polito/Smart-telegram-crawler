import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd
import joblib
import shutil

from bertopic import BERTopic

'''
python 2.1_choose_best_model.py --input 0 --choice coherence --suffix "paraphrase-MiniLM-L6-v2_umap5_umap5_u
map0.0_hdbscan30"
'''

# --------------------
# Parse args
# --------------------
parser = argparse.ArgumentParser(description="Select and refine the best BERTopic model from grid_search results")
parser.add_argument("--input", type=str, default="0", help="level depth (e.g., 0, 1, 2, ...)")
parser.add_argument("--choice", type=str, default="avg_score",
                    choices=["silhouette", "coherence", "diversity", "avg_score"],
                    help="selection criterion for the best model")
parser.add_argument("--suffix", type=str, default=None,
                    help=("Exact model suffix to load, e.g. "
                          "'/paraphrase-MiniLM-L6-v2_umap5_umap5_umap0.0_hdbscan30'. "
                          "If provided, overrides metric-based selection."))
args = parser.parse_args()

level_depth = args.input
choice = args.choice
suffix_arg = args.suffix.strip() if args.suffix else None
if suffix_arg and (suffix_arg.startswith("/") or suffix_arg.startswith("\\")):
    # ripulisci eventuale slash iniziale
    suffix_arg = suffix_arg[1:]

# --------------------
# Base paths
# --------------------
base_dir = f"../results/levels/level_{level_depth}"
grid_dir = os.path.join(base_dir, "grid_search")

grid_search_results_path = os.path.join(grid_dir, f"grid_search_results_level_{level_depth}.csv")
df_sampled_path = os.path.join(grid_dir, f"df_sampled_level_{level_depth}.csv")

bertopic_models_dir = os.path.join(grid_dir, f"bertopic_models_level_{level_depth}")
vectorizers_dir = os.path.join(grid_dir, f"vectorizers_level_{level_depth}")
embeddings_dir = os.path.join(grid_dir, f"embeddings_level_{level_depth}")

final_best_dir = os.path.join(base_dir, "best")
os.makedirs(final_best_dir, exist_ok=True)

best_vectorizer_path   = os.path.join(final_best_dir, "best_vectorizer.pkl")
best_model_pkl_path    = os.path.join(final_best_dir, "best_model.pkl")   # FILE unico richiesto
best_note_path         = os.path.join(final_best_dir, "best_model_note")  # nota testuale
best_note_path2 = os.path.join("./", "best_model_note")

# --------------------
# Pre-flight checks
# --------------------
missing = []
for pth in [grid_search_results_path, df_sampled_path, bertopic_models_dir, embeddings_dir]:
    if not os.path.exists(pth):
        missing.append(pth)

if missing:
    print("Missing required paths/files:")
    for m in missing:
        print("  -", m)
    raise SystemExit(1)

# --------------------
# Load results and sampled dataframe
# --------------------
df_grid = pd.read_csv(grid_search_results_path)
df_sampled = pd.read_csv(df_sampled_path)

required_cols = {
    "model", "umap_n_components", "umap_n_neighbors", "umap_min_dist",
    "hdbscan_min_cluster_size", "silhouette", "coherence", "diversity"
}
if not required_cols.issubset(df_grid.columns):
    raise ValueError(f"Missing columns in grid_search_results; required: {required_cols}")

df_grid["avg_score"] = (df_grid["silhouette"] + df_grid["coherence"] + df_grid["diversity"]) / 3

# --------------------
# Utility: parse suffix
# --------------------
_suffix_re = re.compile(
    r'^(?P<model>.+?)_umap(?P<ncomp>[^_]+)_umap(?P<nneigh>[^_]+)_umap(?P<mindist>[^_]+)_hdbscan(?P<mcs>\d+)$'
)

def parse_suffix(s: str):
    """Return dict with model and params from a suffix string."""
    m = _suffix_re.match(s)
    if not m:
        raise ValueError(f"Suffix not in expected format: {s}")
    gd = m.groupdict()
    return {
        "model": gd["model"],
        "umap_n_components": int(float(gd["ncomp"])),
        "umap_n_neighbors": int(float(gd["nneigh"])),
        "umap_min_dist": float(gd["mindist"]),
        "hdbscan_min_cluster_size": int(gd["mcs"]),
    }

# --------------------
# Decide selection mode (by suffix OR by metric)
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

    mask = (
        (df_grid["model"] == best_model_name) &
        (df_grid["umap_n_components"] == uc["n_components"]) &
        (df_grid["umap_n_neighbors"] == uc["n_neighbors"]) &
        (df_grid["umap_min_dist"] == uc["min_dist"]) &
        (df_grid["hdbscan_min_cluster_size"] == hc["min_cluster_size"])
    )
    if mask.any():
        best_row = df_grid.loc[mask].iloc[0]
    else:
        print("[WARN] Exact suffix row not found in df_grid; proceeding anyway.")
        best_row = pd.Series({
            "model": best_model_name,
            "umap_n_components": uc["n_components"],
            "umap_n_neighbors": uc["n_neighbors"],
            "umap_min_dist": uc["min_dist"],
            "hdbscan_min_cluster_size": hc["min_cluster_size"],
            choice: np.nan
        })
else:
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
# Model path
# --------------------
model_path = Path(os.path.join(bertopic_models_dir, suffix))
if not model_path.exists():
    print(f"Model directory not found: {model_path}")
    raise SystemExit(1)

print(f"Loading model ({'forced suffix' if suffix_arg else 'metric '+choice}) from: {model_path}")
topic_model = BERTopic.load(str(model_path))

# --------------------
# Load embeddings (based on model name)
# --------------------
embeddings_path = os.path.join(embeddings_dir, f"embedding_{best_model_name}_level_{level_depth}.npy")
if not os.path.exists(embeddings_path):
    print(f"Embeddings not found: {embeddings_path}")
    raise SystemExit(1)

embeddings = np.load(embeddings_path)
print("Embeddings shape:", embeddings.shape)

# --------------------
# Text column
# --------------------
if "text_preprocessed" not in df_sampled.columns:
    raise ValueError(f"Text column text_preprocessed not found in df_sampled.")
text_preprocessed_list = df_sampled["text_preprocessed"].tolist()

# --------------------
# Optional: reduce outliers and update topics
# --------------------
try:
    new_topics = topic_model.reduce_outliers(
        text_preprocessed_list, topic_model.topics_, strategy="c-tf-idf", threshold=0.1
    )
    topic_model.update_topics(
        text_preprocessed_list,
        topics=new_topics,
        vectorizer_model=topic_model.vectorizer_model,
        top_n_words=20
    )
    _ = topic_model.get_topic_info()
    print("Outliers reduced and topics updated.")
except Exception as e:
    print(f"Reduce/update topics not executed ({e}). Proceeding with saving the model anyway.")

# --------------------
# Save outputs (FILE unico .pkl, con sovrascrittura pulita)
# --------------------
#Salva/sovrascrivi il vectorizer (file .pkl)
if hasattr(topic_model, "vectorizer_model") and topic_model.vectorizer_model is not None:
    joblib.dump(topic_model.vectorizer_model, best_vectorizer_path)  # sovrascrive se esiste
    print(f"Saved vectorizer to: {best_vectorizer_path}")
else:
    print("Vectorizer not present in the model; skipping vectorizer save.")

#Salva il modello come singolo file .pkl
joblib.dump(topic_model, best_model_pkl_path)
print(f"Best model saved to single file: {best_model_pkl_path}")

#Scrivi/sovrascrivi le note
with open(best_note_path, "w", encoding="utf-8") as f:
    f.write(f"{choice}: {suffix}\n")
print(f"Wrote note: {best_note_path} -> '{choice}: {suffix}'")
with open(best_note_path2, "w", encoding="utf-8") as f:
    f.write(f"{choice}: {suffix}\n")
print(f"Wrote note: {best_note_path2} -> '{choice}: {suffix}'")

print(f"\nDone. Criterion tag: {choice} ({'forced via --suffix' if suffix_arg else 'selected by metric'})")
print(f"Selected suffix: {suffix}")
print(f"Best row (if matched):\n{best_row.to_string()}")
