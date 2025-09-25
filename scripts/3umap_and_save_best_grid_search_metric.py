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


parser = argparse.ArgumentParser(description="Run BERTopic with chosen metric")
parser.add_argument("--choice", type=str, required=True,
                    help="Metric to select best model (e.g., 'coherence', 'silhouette', 'diversity')")
args = parser.parse_args()
choice = args.choice

# Start preprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#getting parameter from command line
parser = argparse.ArgumentParser(description="Preprocess messages for a list of Telegram channels.")
parser.add_argument(
    "--input",
    type=str,
    default="0",
    help="Depth of the hierarchy(default: 0)"
)
args = parser.parse_args()
#input path
level_depth = args.input

# Example structure of "../final/next/next_grid_search_results.csv"
#
# Each row = one experiment (one embedding model + one UMAP config + one HDBSCAN config).
#
# Columns:
#   model                    → which SentenceTransformer model was used
#   umap_n_components        → dimensionality of UMAP output (e.g. 3 or 5)
#   umap_n_neighbors         → neighborhood size used by UMAP
#   umap_min_dist            → minimum distance between UMAP points
#   hdbscan_min_cluster_size → minimum cluster size for HDBSCAN
#   coherence                → topic coherence score (higher = topics make more sense)
#   diversity                → topic diversity score (higher = less overlap between topics)
#   silhouette               → silhouette score of the clustering (higher = better separation)
#   n_outliers               → number of documents marked as outliers (topic = -1)
#   n_topics                 → number of discovered topics (excluding outliers)
#   min_topic                → size of the smallest topic (in number of documents)
#   max_topic                → size of the largest topic (in number of documents)
#
# Example rows (fake numbers for illustration):
#
# model,umap_n_components,umap_n_neighbors,umap_min_dist,hdbscan_min_cluster_size,coherence,diversity,silhouette,n_outliers,n_topics,min_topic,max_topic
# all-distilroberta-v1,5,25,0.0,100,0.423,0.871,0.312,150,12,18,230
# all-distilroberta-v1,3,5,0.1,50,0.401,0.856,0.295,180,10,20,210
# paraphrase-MiniLM-L6-v2,5,125,0.0,500,0.388,0.902,0.276,220,8,15,180
# all-MiniLM-L6-v2,3,25,0.1,250,0.412,0.889,0.301,140,11,19,240



df_grid = pd.read_csv(f"../results/levels/level_{level_depth}/grid_search_results_level_{level_depth}.csv")
df_sampled = pd.read_csv(
    f"../results/levels/level_{level_depth}/grid_search/df_sampled_level_{level_depth}.csv"
)
df_grid['avg_score'] = (df_grid['silhouette'] + df_grid['coherence'] + df_grid['diversity']) / 3
df_grid=df_grid[(df_grid['hdbscan_min_cluster_size']!=500) & (df_grid['hdbscan_min_cluster_size']!=30)]

best_models = {
    'silhouette': df_grid.sort_values(by='silhouette', ascending=False).iloc[0],
    'coherence': df_grid.sort_values(by='coherence', ascending=False).iloc[0],
    'diversity': df_grid.sort_values(by='diversity', ascending=False).iloc[0],
    'avg_score': df_grid.sort_values(by='avg_score', ascending=False).iloc[0]
}

topic_models = {}
for key, row in best_models.items():
    suffix = f"{row['model']_umap{row['umap_n_components']_umap{row['umap_n_neighbors']_umap{row['umap_min_dist']_hdbscan{row['hdbscan_min_cluster_size']}"
    model_path = Path(f"../results/levels/level_{level_depth}/grid_search/bertopic_models_level_{level_depth}/{suffix}")
    Z
    if not model_path.exists():
        print(f"Path not found: {model_path}")
        continue
    
    print(f"Loading model for best {key} from: {model_path}")
    topic_model = BERTopic.load(model_path)
    topic_models[key] = topic_model


# Select the chosen model
best_model = topic_models[choice]

# Create directory for saving the model
final_best_dir = f"../results/levels/level_{level_depth}/best_{level_depth}"
os.makedirs(final_best_dir, exist_ok=True)



# Save the associated vectorizer
best_vectorizer_path = os.path.join(final_best_dir, "best_vectorizer.pkl")
joblib.dump(best_model.vectorizer_model, best_vectorizer_path)

row = best_models[choice]
best_model_name = row['model']

embeddings_path = f"../results/levels/level_{level_depth}/best_{level_depth}/embeddings_{best_model_name}_level_{level_depth}.npy"
embeddings = np.load(embeddings_path)

print("Embeddings shape:", embeddings.shape)

umap_model=best_model.umap_model
reduced_embeddings=umap_model.transform(embeddings)

#Quando BERTopic prova a riassegnare un outlier, confronta il suo contenuto testuale con i topic usando una misura di similarità tra vettori TF-IDF.
#Di solito si tratta di una similarità coseno, che varia tra:
#0.0 = nessuna somiglianza
#1.0 = perfetta somiglianza
#0.1 è molto poco
new_topics = best_model.reduce_outliers(list(df_sampled['text_preprocessed']), best_model.topics_ , strategy="c-tf-idf", threshold=0.1)

# update_topics recalculates the topic representations after reassignments.
# It takes the new topic assignments (including reinserted outliers),
# re-vectorizes the documents using the provided vectorizer_model,
# and rebuilds the c-TF-IDF representation for each topic.
# The 'top_n_words' parameter controls how many of the most representative
# words per topic are stored and updated in the model.
best_model.update_topics(list(df_sampled['text_preprocessed']), topics=new_topics,
                          vectorizer_model=best_model.vectorizer_model,top_n_words=20)
best_model.get_topic_info()

# Create directory for saving the model
os.makedirs(final_best_dir, exist_ok=True)

# Save the BERTopic model
best_model_path = os.path.join(final_best_dir, "best_model.pkl")
best_model.save(best_model_path)
