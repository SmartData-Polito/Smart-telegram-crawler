import argparse
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import shutil

from bertopic import BERTopic

# --------------------
# Arguments
# --------------------
parser = argparse.ArgumentParser(description="Select and refine the best BERTopic model from grid_search results")
parser.add_argument("--input", type=str, default="0", help="level depth (e.g., 0, 1, 2, ...)")
parser.add_argument("--choice", type=str, default="avg_score",
                    choices=["silhouette", "coherence", "diversity", "avg_score"],
                    help="selection criterion for the best model")
args = parser.parse_args()

level_depth = args.input
choice = args.choice

# --------------------
# Base paths (aligned with your grid_search script)
# --------------------
base_dir = f"../results/levels/level_{level_depth}"
grid_dir = os.path.join(base_dir, "grid_search")

grid_search_results_path = os.path.join(grid_dir, f"grid_search_results_level_{level_depth}.csv")
df_sampled_path = os.path.join(grid_dir, f"df_sampled_level_{level_depth}.csv")

bertopic_models_dir = os.path.join(grid_dir, f"bertopic_models_level_{level_depth}")
vectorizers_dir = os.path.join(grid_dir, f"vectorizers_level_{level_depth}")
embeddings_dir = os.path.join(grid_dir, f"embeddings_level_{level_depth}")

final_best_dir = os.path.join(base_dir, f"best")
os.makedirs(final_best_dir, exist_ok=True)

best_vectorizer_path = os.path.join(final_best_dir, "best_vectorizer.pkl")
best_model_path = os.path.join(final_best_dir, "best_model")  # directory, not .pkl

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

# Compute avg_score based on the three metrics
required_cols = {
    "model", "umap_n_components", "umap_n_neighbors", "umap_min_dist",
    "hdbscan_min_cluster_size", "silhouette", "coherence", "diversity"
}
if not required_cols.issubset(df_grid.columns):
    raise ValueError(f"Missing columns in grid_search_results; required: {required_cols}")

df_grid["avg_score"] = (df_grid["silhouette"] + df_grid["coherence"] + df_grid["diversity"]) / 3

# Select the best row by the requested criterion
best_row = df_grid.sort_values(by=choice, ascending=False).iloc[0]
best_model_name = best_row["model"]
uc = {
    "n_components": int(best_row["umap_n_components"]),
    "n_neighbors": int(best_row["umap_n_neighbors"]),
    "min_dist": float(best_row["umap_min_dist"]),
}
hc = {
    "min_cluster_size": int(best_row["hdbscan_min_cluster_size"])
}

# --------------------
# Reconstruct the model directory name exactly as saved by grid_search
# --------------------
suffix = (
    f"{best_model_name}_"
    f"umap{uc['n_components']}_"
    f"umap{uc['n_neighbors']}_"
    f"umap{uc['min_dist']}_"
    f"hdbscan{hc['min_cluster_size']}"
)

model_path = Path(os.path.join(bertopic_models_dir, suffix))
if not model_path.exists():
    print(f"Model directory not found: {model_path}")
    raise SystemExit(1)

print(f"Loading best model ({choice}) from: {model_path}")
topic_model = BERTopic.load(str(model_path))

# --------------------
# Load embeddings consistent with best_model_name
# --------------------
embeddings_path = os.path.join(embeddings_dir, f"embedding_{best_model_name}_level_{level_depth}.npy")
if not os.path.exists(embeddings_path):
    print(f"Embeddings not found: {embeddings_path}")
    raise SystemExit(1)

embeddings = np.load(embeddings_path)
print("Embeddings shape:", embeddings.shape)

# --------------------
# Text column consistent with grid_search
# --------------------
text_col = "text_preprocessed"
if text_col not in df_sampled.columns:
    raise ValueError(f"Text column '{text_col}' not found in df_sampled.")
docs = df_sampled[text_col].tolist()

# --------------------
# Optional: reduce outliers and update topics
# Note: topic_model.topics_ was set during fit in grid_search
# --------------------
try:
    new_topics = topic_model.reduce_outliers(
        docs, topic_model.topics_, strategy="c-tf-idf", threshold=0.1
    )
    topic_model.update_topics(
        docs,
        topics=new_topics,
        vectorizer_model=topic_model.vectorizer_model,
        top_n_words=20
    )
    _ = topic_model.get_topic_info()  # force topic info refresh
    print("Outliers reduced and topics updated.")
except Exception as e:
    print(f"Reduce/update topics not executed ({e}). Proceeding with saving the model anyway.")

# --------------------
# Save outputs: vectorizer (pkl) and model (directory)
# --------------------
if hasattr(topic_model, "vectorizer_model") and topic_model.vectorizer_model is not None:
    joblib.dump(topic_model.vectorizer_model, best_vectorizer_path)
    print(f"Saved vectorizer to: {best_vectorizer_path}")
else:
    print("Vectorizer not present in the model; skipping vectorizer save.")

# Ensure a clean target directory for the best model
if os.path.exists(best_model_path):
    shutil.rmtree(best_model_path, ignore_errors=True)

topic_model.save(best_model_path)
print(f"Best model saved to: {best_model_path}")

print(f"\nDone. Criterion: {choice}")
print(f"Best row:\n{best_row.to_string()}")
