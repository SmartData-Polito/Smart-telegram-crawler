import pandas as pd
import torch
import time
from bertopic import BERTopic
from cuml.cluster import HDBSCAN
from cuml.manifold import UMAP
from sentence_transformers import SentenceTransformer
from sklearn.metrics import silhouette_score
import os
from tqdm import tqdm
from glob import glob
from octis.evaluation_metrics.diversity_metrics import TopicDiversity
from octis.evaluation_metrics.coherence_metrics import Coherence
import numpy as np


# Step 1: Load and preprocess the data
df = pd.concat(([pd.read_csv(file, sep='\t') for file in tqdm(glob('sample/*.csv'))]), ignore_index=True)
df=df.iloc[:len(df)//10]

texts=[sentence.split() for sentence in df['processed_text'].to_list()]
'''
texts = [
    ["i", "love", "machine", "learning"],
    ["data", "science", "is", "fun"],
    ["deep", "learning", "models", "are", "powerful"]
]*/
'''
def get_metrics(topic_model,texts=texts):
    '''
    Return: diversity_score,coherence_score
    '''
    topics = topic_model.get_topics()
    '''
    topics={
    0: [('apple', 0.3), ('banana', 0.2), ...],
    1: [('data', 0.4), ('science', 0.3), ...],
    ...
    }
    '''
    topics_list = []
    for topic_id, topic_words in topics.items():
        '''
        topic_words=[('apple', 0.3), ('banana', 0.2), ...]
        '''
        if topic_id!=-1:
            # Take words for each topic where >= 10
            words=[word[0] for word in topic_words if word[0]!='']
            if len(words)>=10:
                topics_list.append(words)  # Extracting only words from (word, probability)

    # Wrap the topics into the expected format
    model_output = {"topics": topics_list}

    '''
    model_output = {
    "topics": [
        ["apple", "banana", ...],
        ["data", "science", ...],
        ...
    ]
    }
    '''
    

    # Now calculate diversity using the correct format
    topic_diversity = TopicDiversity(topk=10)  # Specify how many top words you want to consider
    diversity_score = topic_diversity.score(model_output)  # Pass the wrapped topics

    # Calculate coherence score
    coherence_metric = Coherence(topk=10,texts=texts)  # Specify top_n for coherence calculation
    coherence_score = coherence_metric.score(model_output)  # Pass the wrapped topics
    print(coherence_score)
    return diversity_score,coherence_score

# Step 2: Set device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)
# List of SentenceTransformer models
models = {
    'all-distilroberta-v1': SentenceTransformer('all-distilroberta-v1'),
    'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2')
}

# Step 3: Precompute embeddings for each model


# Step 4: Define grid search for UMAP and HDBSCAN parameters
umap_params = [
    {'n_components': 5, 'n_neighbors': 5, 'min_dist': 0.0},
    {'n_components': 5, 'n_neighbors': 25, 'min_dist': 0.0},
    {'n_components': 5, 'n_neighbors': 125, 'min_dist': 0.0},
    
    {'n_components': 5, 'n_neighbors': 5, 'min_dist': 0.1},
    {'n_components': 5, 'n_neighbors': 25, 'min_dist': 0.1},
    {'n_components': 5, 'n_neighbors': 125, 'min_dist': 0.1},
]

hdbscan_params = [
    {'min_cluster_size': 100},
    {'min_cluster_size': 500}
     ]

# Step 5: Check if results already exist
if os.path.exists('grid_search_results.csv'):
    results_df = pd.read_csv('grid_search_results.csv')
else:
    results_df = pd.DataFrame(columns=['model',
                'umap_n_components',
                'umap_n_neighbors',
                'umap_min_dist',
                'hdbscan_min_cluster_size',
                'coherence',
                'diversity',
                'silhouette'])

# Initialize the results list
results = []

# Step 6: Perform grid search
best_score = -1
best_model = None
best_params = None

'''
models = {
    'all-distilroberta-v1': SentenceTransformer('all-distilroberta-v1'),
    'paraphrase-MiniLM-L6-v2': SentenceTransformer('paraphrase-MiniLM-L6-v2'),
    'all-MiniLM-L6-v2': SentenceTransformer('all-MiniLM-L6-v2')
}

umap_params = [
    {'n_components': 3, 'n_neighbors': 5, 'min_dist': 0.0},
    {'n_components': 3, 'n_neighbors': 25, 'min_dist': 0.0},
]
'''
for model_name, model_instance in tqdm(models.items()):
    # for every possible model of Sentence Transformer
    if os.path.exists(f'embeddings/embeddings_{model_name}.npy'):
        embeddings = np.load(f'embeddings/embeddings_{model_name}.npy')
        print(f'embedding {model_name} loaded')
    else:
        model_instance = model_instance.to(device)
        embeddings = model_instance.encode(df['processed_text'].tolist(), show_progress_bar=True, device=device)
        np.save(f'embeddings/embeddings_{model_name}.npy', embeddings)
    
    for umap_config in umap_params:
        for hdbscan_config in hdbscan_params:
            # Check if this combination has already been run
            '''
                results_df = pd.DataFrame(columns=[
                'model',
                'umap_n_components',
                'umap_n_neighbors',
                'umap_min_dist',
                'hdbscan_min_cluster_size',
                'coherence',
                'diversity',
                'silhouette'])
            '''
            if ((len(results_df)>0) and
                ((results_df['model'] == model_name) &
                (results_df['umap_n_components'] == umap_config['n_components']) &
                (results_df['umap_n_neighbors'] == umap_config['n_neighbors']) &
                (results_df['umap_min_dist'] == umap_config['min_dist']) &
                (results_df['hdbscan_min_cluster_size'] == hdbscan_config['min_cluster_size'])).any()):
                print(f"Skipping already tested configuration: {model_name} with UMAP {umap_config} and HDBSCAN {hdbscan_config}")
                continue

            # Initialize UMAP and HDBSCAN models with current parameters
            umap_model = UMAP(**umap_config)
            hdbscan_model = HDBSCAN(**hdbscan_config)
            
            # Step 7: Apply UMAP and HDBSCAN to the embeddings
            t0 = time.time()
            topic_model = BERTopic(
                embedding_model=None,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                verbose=True,
                top_n_words=20,
                language = 'english',
            )
            
            topics, probs = topic_model.fit_transform(df['processed_text'],embeddings=embeddings)
            print(f"Execution time for {model_name} UMAP: {time.time()-t0}s")
            
            topics=np.array(topics)
            
            
            # Step 8: Save the model
            model_filename = f'bertopic_models/{model_name}_umap{umap_config["n_components"]}_umap{umap_config["n_neighbors"]}_umap{umap_config["min_dist"]}_hdbscan{hdbscan_config["min_cluster_size"]}.pkl'
            topic_model.save(model_filename)

            # Step 9: Compute evaluation metrics
            diversity,coherence=get_metrics(topic_model)
            
            
            umap_model = topic_model.umap_model
            reduced_embedding=umap_model.transform(embeddings)
            topics=np.array(topic_model.topics_)
            silhouette=silhouette_score(reduced_embedding[topics!=-1],topics[topics!=-1])
            
            # Store the best model based on silhouette score
            avg_score = (coherence + diversity + silhouette) / 3
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = topic_model
                best_params = {'model_name': model_name, 'umap_config': umap_config, 'hdbscan_config': hdbscan_config}
                print(best_params)

           
            topics=pd.Series(topics)
            
            # Step 10: Log the results
            results.append({
                'model': model_name,
                'umap_n_components': umap_config['n_components'],
                'umap_n_neighbors': umap_config['n_neighbors'],
                'umap_min_dist': umap_config['min_dist'],
                'hdbscan_min_cluster_size': hdbscan_config['min_cluster_size'],
                'coherence': coherence,
                'diversity': diversity,
                'silhouette': silhouette,
                'n_outliers':(topics==-1).sum(),
                'n_topics':topics.nunique()-1,
                'min_topic':topics.value_counts().min(),
                'max_topic':topics.value_counts().max(),
                
                })
            print(results[-1])
                
                
            pd.concat([results_df, pd.DataFrame(results)], ignore_index=True).to_csv('grid_search_results.csv', index=False)
            
                
# Step 12: Output the best parameters and score
print(f"Best Parameters: {best_params}")
print(f"Best Average Score: {best_score}")




