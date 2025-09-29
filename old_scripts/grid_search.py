#from cuml.cluster import HDBSCAN     # se vorrai usare versione GPU, altrimenti commenta
#from cuml.manifold import UMAP       # se vorrai usare versione GPU, altrimenti commenta
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
from bertopic import BERTopic
print("debug")
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
from octis.evaluation_metrics.coherence_metrics import Coherence
from joblib import Parallel, delayed
import random
import os
import time
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import joblib
# Fare preprocessing dei testi:
import os
import re
from typing import Callable, Union
# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm
from unidecode import unidecode
import langdetect
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
from pathlib import Path
from bertopic import BERTopic
import pandas as pd
import plotly.io as pio


SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Step 1: sample & save
#here we are working only with the seeds
output_path_preprocessed_english_messages = "../material/preprocessed_english_messages.tsv.gz"
df_english_preprocessed_non_empty_channels = pd.read_csv(output_path_preprocessed_english_messages, sep='\t', compression='gzip')
df_sampled = df_english_preprocessed_non_empty_channels.sample(frac=1, random_state=SEED)
df_sampled.to_csv("../final/df_sampled.csv", index=False)

print("imported df_sampled")

texts = [sentence.split() for sentence in df_sampled['text_preprocessed'].tolist()]

def get_metrics(topic_model, texts=texts):
    """
    Calcola diversity e coherence del topic model.
    Ritorna: diversity_score, coherence_score
    """
    topics = topic_model.get_topics()
    
    topics_list = []
    for topic_id, topic_words in topics.items():
        if topic_id == -1:
            continue
        words = [word[0] for word in topic_words if word[0] != '']
        if len(words) >= 10:
            topics_list.append(words)

    if not topics_list:
        print("⚠️ Nessun topic con almeno 10 parole trovato.")
        return 0.0, 0.0

    filtered = [t for t in topics_list if len(t) >= 10]
    if not filtered:
        print("⚠️ Tutti i topic hanno meno di 10 parole.")
        return 0.0, 0.0

    diversity_score = TopicDiversity(topk=10).score({"topics": filtered})
    coherence_score = Coherence(topk=10, texts=texts).score({"topics": filtered})
    return diversity_score, coherence_score

# Step 2: Set device & models
os.environ["TOKENIZERS_PARALLELISM"] = "true"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

models = {
    'all-distilroberta-v1': SentenceTransformer('all-distilroberta-v1'),
    'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2')
}

# Step 3: vectorizer (unused in BERTopic instantiation here, but kept)
from sklearn.feature_extraction.text import CountVectorizer
vectorizer_model = CountVectorizer(ngram_range=(1,1), stop_words="english")

# Step 4: grid params
umap_params = [
    {'n_components': 5, 'n_neighbors': 5,   'min_dist': 0.0},
    {'n_components': 5, 'n_neighbors': 25,  'min_dist': 0.0},
    {'n_components': 5, 'n_neighbors': 125, 'min_dist': 0.0},
    {'n_components': 5, 'n_neighbors': 5,   'min_dist': 0.1},
    {'n_components': 5, 'n_neighbors': 25,  'min_dist': 0.1},
    {'n_components': 5, 'n_neighbors': 125, 'min_dist': 0.1},
    {'n_components': 3, 'n_neighbors': 5,   'min_dist': 0.0},
    {'n_components': 3, 'n_neighbors': 25,  'min_dist': 0.0},
    {'n_components': 3, 'n_neighbors': 125, 'min_dist': 0.0},
    {'n_components': 3, 'n_neighbors': 5,   'min_dist': 0.1},
    {'n_components': 3, 'n_neighbors': 25,  'min_dist': 0.1},
    {'n_components': 3, 'n_neighbors': 125, 'min_dist': 0.1},
]

hdbscan_params = [
    {'min_cluster_size': 30},
    {'min_cluster_size': 50},
    {'min_cluster_size': 100},
    {'min_cluster_size': 250},
    {'min_cluster_size': 500},
]

# Step 5: prepare output CSV
out_path = '../final/grid_search_results.csv'
cols = [
    'model','umap_n_components','umap_n_neighbors','umap_min_dist',
    'hdbscan_min_cluster_size','coherence','diversity','silhouette',
    'n_outliers','n_topics','min_topic','max_topic'
]
if not os.path.exists(out_path):
    pd.DataFrame(columns=cols).to_csv(out_path, index=False)

# helper to check existing runs
def not_already_tested(model_name, uc, hc):
    df = pd.read_csv(out_path)
    return not ((df['model'] == model_name) &
                (df['umap_n_components'] == uc['n_components']) &
                (df['umap_n_neighbors'] == uc['n_neighbors']) &
                (df['umap_min_dist'] == uc['min_dist']) &
                (df['hdbscan_min_cluster_size'] == hc['min_cluster_size'])).any()

# define single-run function
def run_single_run(model_name, embeddings, umap_config, hdbscan_config):
    umap_model    = UMAP(**umap_config, random_state=SEED)
    hdbscan_model = HDBSCAN(**hdbscan_config)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        verbose=False,
        top_n_words=20,
        language='english'
    )
    topics, _ = topic_model.fit_transform(df_sampled['text_preprocessed'], embeddings=embeddings)

    diversity, coherence = get_metrics(topic_model)
    reduced = umap_model.transform(embeddings)
    labels  = np.array(topic_model.topics_)
    mask    = labels != -1
    sil = (silhouette_score(reduced[mask], labels[mask])
           if mask.sum() > 1 and len(np.unique(labels[mask])) > 1 else 0.0)

    suffix = f"{model_name}_umap{umap_config['n_components']}_umap{umap_config['n_neighbors']}_umap{umap_config['min_dist']}_hdbscan{hdbscan_config['min_cluster_size']}"
    model_dir = "../final/bertopic_models"
    os.makedirs(model_dir, exist_ok=True)
    topic_model.save(os.path.join(model_dir, suffix))  # salva come directory
    vectorizer_dir = "../final/vectorizers"
    os.makedirs(vectorizer_dir, exist_ok=True)
    joblib.dump(topic_model.vectorizer_model, os.path.join(vectorizer_dir, f"vectorizer_{suffix}.pkl"))

    series = pd.Series(labels[labels != -1])
    return {
        'model': model_name,
        'umap_n_components': umap_config['n_components'],
        'umap_n_neighbors':  umap_config['n_neighbors'],
        'umap_min_dist':      umap_config['min_dist'],
        'hdbscan_min_cluster_size': hdbscan_config['min_cluster_size'],
        'coherence':   coherence,
        'diversity':   diversity,
        'silhouette':  sil,
        'n_outliers':  int((labels == -1).sum()),
        'n_topics':    int(len(np.unique(labels)) - ( -1 in labels )),
        'min_topic':   int(series.value_counts().min()),
        'max_topic':   int(series.value_counts().max())
    }

for model_name, model_instance in tqdm(models.items()):
    emb_path = f'../final/final_embeddings_{model_name}.npy'
    if os.path.exists(emb_path):
        print(f"Embedding già esistente: {model_name}")
        continue
    model_instance = model_instance.to(device)
    embeddings = model_instance.encode(
        df_sampled['text_preprocessed'].tolist(),
        show_progress_bar=True,
        device=device
    )
    np.save(emb_path, embeddings)
    print(f"Salvato: {emb_path}")

for model_name, model_instance in tqdm(models.items()):
    emb_path = f'../final/final_embeddings_{model_name}.npy'
    if not os.path.exists(emb_path):
        print(f"[⚠️] Manca embedding per {model_name}, skippato.")
        continue
    embeddings = np.load(emb_path)

    # build param grid for this model
    param_grid = [
        (uc, hc)
        for uc in umap_params
        for hc in hdbscan_params
        if not_already_tested(model_name, uc, hc)
    ]

    # run in parallel
    batch_results = Parallel(n_jobs=os.cpu_count() - 1, verbose=5)(
        delayed(run_single_run)(model_name, embeddings, uc, hc)
        for uc, hc in param_grid
    )

    # append to CSV
    temp_df = pd.DataFrame(batch_results)
    temp_df.to_csv(out_path, mode='a', index=False, header=False)