import numpy as np
from sklearn.manifold import trustworthiness
from sklearn.model_selection import ParameterGrid
import pandas as pd
from tqdm import tqdm

from cuml.manifold import UMAP

from glob import glob
import numpy as np
from tqdm import tqdm 

vectors = np.load('03_LLama_Summarization/all_vectors.npz')
embeddings = np.stack([vectors[key] for key in tqdm(vectors.files)]) #Long (~12 min)


embeddings=(embeddings-embeddings.mean(axis=0))/embeddings.std(axis=0)


param_grid = {
    'n_neighbors': [10,50],
    'min_dist': [0, 0.1, 0.3],
    'metric': ['cosine','euclidean'],
    'n_components': [5, 10]
}
grid = list(ParameterGrid(param_grid))
results = []

for params in tqdm(grid):
    reducer = UMAP(**params, random_state=42)
    red_embedding = reducer.fit_transform(embeddings)  #  shape: (100000, 4000)
    trust = trustworthiness(embeddings[::3], red_embedding[::3], n_neighbors=params['n_neighbors'])
    results.append({**params, 'trustworthiness': trust})
    print(f"Params: {params}, Trustworthiness: {trust:.4f}")
    
df_results = pd.DataFrame(results)
df_old_results=pd.read_csv('UMAP_parameter_selection.csv.gz', compression='gzip')

df_results.to_csv('UMAP_parameter_selection.csv.gz', compression='gzip', index=False)