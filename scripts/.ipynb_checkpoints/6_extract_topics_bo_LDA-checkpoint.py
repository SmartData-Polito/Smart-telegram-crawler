#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per estrarre e salvare i topic del modello LDA ottimale.
Uso: python 6_extract_topics.py --level 0 [--topn 20]

docs = [
    "cane corre veloce",
    "gatto dorme spesso",
    "cane dorme"
]
X_docs = vectorizer.transform(docs)
doc	cane	corre	veloce	gatto	dorme	spesso
0	1	1	1	0	0	0
1	0	0	0	1	1	1
2	1	0	0	0	1	0


doc_topic = lda_model.transform(X_docs) si ha:
[
  [0.90, 0.10],   # doc0 → quasi tutto topic 0 (politica)
  [0.05, 0.95],   # doc1 → quasi tutto topic 1 (crypto)
  [0.80, 0.20],   # doc2 → topic 0
  [0.20, 0.80],   # doc3 → topic 1
]

shape = (n_topics, n_terms)
riga k → distribuzione delle parole del topic k
(la β_k del modello)
array([
    [ 9.8,  8.1, 10.5,  0.3,  0.2,  0.1],   # Topic 0
    [ 0.4,  0.6,  0.2, 12.1, 14.8, 10.7],   # Topic 1
    [ 5.0,  4.5,  5.2,  4.9,  4.8,  4.7]    # Topic 2
])

"""

import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd

SEED = 42  # per avere random reproducible

def main():
    parser = argparse.ArgumentParser(description="Estrae topic, parole chiave e documenti dal modello LDA")
    parser.add_argument("--level", type=str, required=True, help="Livello di profondità (es. 0, 1, 2)")
    parser.add_argument("--topn", type=int, default=20, help="Numero di parole chiave per topic (default: 20)")
    args = parser.parse_args()
    
    level_depth = args.level
    topn = args.topn
    
    # Percorsi base
    base_root = f"../results/levels/level_{level_depth}/grid_search_lda_optimized/"
    best_k_path = os.path.join(base_root, f"optuna_best_k_level_{level_depth}.json")
    output_txt = os.path.join(".", f"6_topics_keywords_level_{level_depth}.txt")

    # File con i messaggi (preprocessing)
    docs_path = (
        f"../results/levels/level_{level_depth}/preProcessing/"
        f"preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz"
    )

    # 1) Verifica esistenza metadata best_k
    if not os.path.exists(best_k_path):
        print(f"ERRORE: File best_k non trovato: {best_k_path}")
        print("Esegui prima lo script di training LDA (grid_search_lda_optimized).")
        return
    
    # 2) Carica metadata best k
    with open(best_k_path, "r", encoding="utf-8") as f:
        best_k_info = json.load(f)
    
    best_k = best_k_info["best_k"]
    print(f"Best K trovato: {best_k}")
    
    # 3) Trova il modello corrispondente
    dir_models = f"{base_root}lda_models_level_{level_depth}/"
    dir_vectorizers = f"{base_root}vectorizers_level_{level_depth}/"
    
    model_files = [f for f in os.listdir(dir_models) if f.startswith(f"LDA_k{best_k}_") and f.endswith(".joblib")]
    if not model_files:
        print(f"ERRORE: Nessun modello trovato per k={best_k} in {dir_models}")
        return
    
    model_path = os.path.join(dir_models, model_files[0])
    print(f"Caricamento modello: {model_path}")
    lda_model = joblib.load(model_path)
    
    # 4) Trova e carica il vectorizer corrispondente
    suffix = model_files[0].replace("LDA_", "").replace(".joblib", "")
    vect_files = [f for f in os.listdir(dir_vectorizers) if suffix in f and f.endswith(".joblib")]
    
    if not vect_files:
        print(f"ERRORE: Vectorizer non trovato per suffix: {suffix}")
        return
    
    vectorizer_path = os.path.join(dir_vectorizers, vect_files[0])
    print(f"Caricamento vectorizer: {vectorizer_path}")
    vectorizer = joblib.load(vectorizer_path)
    vocab = vectorizer.get_feature_names_out()
    
    # 5) Carico i documenti preprocessati per estrarre i top docs
    if not os.path.exists(docs_path):
        print(f"ERRORE: File con i messaggi non trovato: {docs_path}")
        return

    print(f"Caricamento documenti da: {docs_path}")
    # carico solo la colonna llm_text_preprocessed per risparmiare memoria
    df_docs = pd.read_csv(docs_path, sep="\t", compression="gzip", usecols=["llm_text_preprocessed"])
    print(f"df_docs.columns: {df_docs.columns}")
    print(f"df_docs[llm_text_preprocessed].head(10): {df_docs['llm_text_preprocessed'].head(10)}")
    df_docs = df_docs[df_docs["llm_text_preprocessed"].astype(str).str.strip() != ""]
    df_docs["llm_text_preprocessed"] = df_docs["llm_text_preprocessed"].astype(str)
    docs = df_docs["llm_text_preprocessed"].tolist()
    n_docs = len(docs)
    print(f"Numero documenti caricati: {n_docs}")

    # 6) BoW/TF con lo stesso vectorizer del training
    print("Trasformazione dei documenti con il vectorizer (X_docs = vectorizer.transform)...")
    X_docs = vectorizer.transform(docs)  # stessa vocab del training

    # 7) Matrice doc-topic
    print("Calcolo matrice doc-topic (doc_topic = lda_model.transform(X_docs))...")
    doc_topic = lda_model.transform(X_docs)  # shape: (n_docs, n_topics)
    n_topics = doc_topic.shape[1]
    print(f"doc_topic shape: {doc_topic.shape}")

    # check di coerenza: n_topics dovrebbe essere == best_k
    if n_topics != best_k:
        print(f"ATTENZIONE: n_topics={n_topics} ma best_k={best_k} (non coincidono!)")

    # Pre-calcolo: topic dominante per ogni documento (per i random)
    dominant_topic = np.argmax(doc_topic, axis=1)  # shape: (n_docs,)

    # 8) Estrai topic → parole
    print(f"\nEstrazione top-{topn} parole per {best_k} topic...")
    W = lda_model.components_  # shape: (n_topics, n_terms)
    
    # 9) Scrivi file di output
    print(f"\nSalvataggio in: {output_txt}")
    rng = np.random.default_rng(SEED)  # RNG per i doc random

    n_top_docs = 3  # numero di documenti più rappresentativi
    n_random_docs = 3  # numero di documenti random per topic

    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Level {level_depth} - {best_k} topics - Top {topn} keywords\n\n")

        for topic_idx in range(W.shape[0]):
            # ---- TOP parole ----
            top_indices = np.argsort(W[topic_idx])[::-1][:topn]
            top_words = [vocab[i] for i in top_indices]

            f.write(f"Topic {topic_idx}: {', '.join(top_words)}\n")

            # ---- TOP 3 documenti per questo topic ----
            topic_column = doc_topic[:, topic_idx]  # shape: (n_docs,)
            # ordino in modo decrescente per prendere i più rappresentativi
            sorted_doc_idx = np.argsort(topic_column)[::-1]
            top_doc_idx = sorted_doc_idx[:min(n_top_docs, n_docs)]

            f.write(f"  Top {len(top_doc_idx)} documents:\n")
            for rank, doc_id in enumerate(top_doc_idx, start=1):
                score = topic_column[doc_id]
                snippet = docs[doc_id][:200].replace("\n", " ")
                f.write(f"    {rank}) doc {doc_id} (score={score:.3f}): {snippet}\n")

            # ---- 3 documenti random "di" questo topic (topic dominante) ----
            # candidati: documenti il cui topic dominante è topic_idx
            candidate_idx = np.where(dominant_topic == topic_idx)[0]

            # evito di ripetere i top doc se possibile
            candidate_idx = np.setdiff1d(candidate_idx, top_doc_idx, assume_unique=False)

            if candidate_idx.size == 0:
                # fallback: se non ci sono doc con topic_idx dominante, pesco da tutti
                candidate_idx = np.setdiff1d(np.arange(n_docs), top_doc_idx, assume_unique=False)

            if candidate_idx.size > 0:
                n_to_sample = min(n_random_docs, candidate_idx.size)
                random_idx = rng.choice(candidate_idx, size=n_to_sample, replace=False)
            else:
                random_idx = np.array([], dtype=int)

            f.write(f"  {len(random_idx)} random documents (topic-dominant if possible):\n")
            for rank, doc_id in enumerate(random_idx, start=1):
                score = topic_column[doc_id]
                snippet = docs[doc_id][:200].replace("\n", " ")
                f.write(f"    {rank}) doc {doc_id} (score={score:.3f}): {snippet}\n")

            f.write("\n")  # riga vuota tra topic e topic
    
    print(f"✓ File creato: {len(W)} topic salvati con {topn} parole ciascuno, "
          f"con {n_top_docs} top docs e {n_random_docs} random docs per topic")

if __name__ == "__main__":
    main()
