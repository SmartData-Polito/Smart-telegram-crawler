import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN

from sklearn.metrics import silhouette_score
from tqdm import tqdm


# --------------------
# Argument parsing
# --------------------
parser = argparse.ArgumentParser(description="Run BERTopic with chosen metric and input level")
parser.add_argument("--choice", type=str, required=True,
                    help="Metric to select best model (e.g., 'coherence', 'silhouette', 'diversity')")
parser.add_argument("--input", type=str, default="0",
                    help="Depth of the hierarchy (default: 0)")
args = parser.parse_args()

choice = args.choice
level_depth = args.input

# --------------------
# Paths and directories
# --------------------
base_results_dir = f"../results/levels/level_{level_depth}"

# Input paths
grid_search_results_path = os.path.join(base_results_dir, "grid_search", f"grid_search_results_level_{level_depth}.csv")
df_sampled_path = os.path.join(base_results_dir, "grid_search", f"df_sampled_level_{level_depth}.csv")
bertopic_models_dir = os.path.join(base_results_dir, "grid_search", f"bertopic_models_level_{level_depth}")

# Output directories
final_best_dir = os.path.join(base_results_dir, f"best_{level_depth}")
os.makedirs(final_best_dir, exist_ok=True)

# Output paths
best_vectorizer_path = os.path.join(final_best_dir, "best_vectorizer.pkl")
best_model_path = os.path.join(final_best_dir, "best_model.pkl")

# --------------------
# Start preprocessing
# --------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# --------------------
# Load results from grid search
# --------------------
df_grid = pd.read_csv(grid_search_results_path)
df_sampled = pd.read_csv(df_sampled_path)

df_grid['avg_score'] = (df_grid['silhouette'] + df_grid['coherence'] + df_grid['diversity']) / 3
df_grid = df_grid[(df_grid['hdbscan_min_cluster_size'] != 500) & (df_grid['hdbscan_min_cluster_size'] != 30)]

best_models = {
    'silhouette': df_grid.sort_values(by='silhouette', ascending=False).iloc[0],
    'coherence': df_grid.sort_values(by='coherence', ascending=False).iloc[0],
    'diversity': df_grid.sort_values(by='diversity', ascending=False).iloc[0],
    'avg_score': df_grid.sort_values(by='avg_score', ascending=False).iloc[0]
}

# --------------------
# Load best BERTopic models
# --------------------
topic_models = {}
for key, row in best_models.items():
    suffix = f"{row['model']}_umap{row['umap_n_components']}_umap{row['umap_n_neighbors']}_umap{row['umap_min_dist']}_hdbscan{row['hdbscan_min_cluster_size']}"
    model_path = Path(os.path.join(bertopic_models_dir, suffix))
    if not model_path.exists():
        print(f"Path not found: {model_path}")
        continue
    
    print(f"Loading model for best {key} from: {model_path}")
    topic_model = BERTopic.load(model_path)
    topic_models[key] = topic_model

# --------------------
# Select the chosen model
# --------------------
best_model = topic_models[choice]

# Save the associated vectorizer
joblib.dump(best_model.vectorizer_model, best_vectorizer_path)

row = best_models[choice]
best_model_name = row['model']

embeddings_path = os.path.join(final_best_dir, f"embeddings_{best_model_name}_level_{level_depth}.npy")
embeddings = np.load(embeddings_path)

print("Embeddings shape:", embeddings.shape)

# --------------------
# Reduce embeddings and reassign outliers
# --------------------
umap_model = best_model.umap_model
reduced_embeddings = umap_model.transform(embeddings)

new_topics = best_model.reduce_outliers(
    list(df_sampled['text_preprocessed']),
    best_model.topics_,
    strategy="c-tf-idf",
    threshold=0.1
)

best_model.update_topics(
    list(df_sampled['text_preprocessed']),
    topics=new_topics,
    vectorizer_model=best_model.vectorizer_model,
    top_n_words=20
)

best_model.get_topic_info()

# --------------------
# Save the BERTopic model
# --------------------
best_model.save(best_model_path)
