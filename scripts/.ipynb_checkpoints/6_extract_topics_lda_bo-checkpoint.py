#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per estrarre e salvare i topic del modello LDA ottimale.
Uso: python 6_extract_topics.py --level 0 [--topn 20]
"""

import os
import argparse
import joblib
import json
import numpy as np

def main():
    parser = argparse.ArgumentParser(description="Estrae topic e parole chiave dal modello LDA")
    parser.add_argument("--level", type=str, required=True, help="Livello di profondità (es. 0, 1, 2)")
    parser.add_argument("--topn", type=int, default=20, help="Numero di parole chiave per topic (default: 20)")
    args = parser.parse_args()
    
    level_depth = args.level
    topn = args.topn
    
    # Percorsi
    base_root = f"../results/levels/level_{level_depth}/grid_search_lda_optimized/"
    best_k_path = os.path.join(base_root, f"optuna_best_k_level_{level_depth}.json")
    output_txt = os.path.join(".", f"6_topics_keywords_level_{level_depth}.txt")
    
    # Verifica esistenza metadata
    if not os.path.exists(best_k_path):
        print(f"ERRORE: File best_k non trovato: {best_k_path}")
        print("Esegui prima lo script di training LDA.")
        return
    
    # Carica metadata best k
    with open(best_k_path, "r", encoding="utf-8") as f:
        best_k_info = json.load(f)
    
    best_k = best_k_info["best_k"]
    print(f"Best K trovato: {best_k}")
    
    # Trova il modello corrispondente
    dir_models = f"{base_root}lda_models_level_{level_depth}/"
    dir_vectorizers = f"{base_root}vectorizers_level_{level_depth}/"
    
    # Cerca il file del modello con best_k
    model_files = [f for f in os.listdir(dir_models) if f.startswith(f"LDA_k{best_k}_") and f.endswith(".joblib")]
    
    if not model_files:
        print(f"ERRORE: Nessun modello trovato per k={best_k} in {dir_models}")
        return
    
    model_path = os.path.join(dir_models, model_files[0])
    print(f"Caricamento modello: {model_path}")
    
    # Carica modello
    lda_model = joblib.load(model_path)
    
    # Trova e carica il vectorizer corrispondente
    suffix = model_files[0].replace("LDA_", "").replace(".joblib", "")
    vect_files = [f for f in os.listdir(dir_vectorizers) if suffix in f and f.endswith(".joblib")]
    
    if not vect_files:
        print(f"ERRORE: Vectorizer non trovato per suffix: {suffix}")
        return
    
    vectorizer_path = os.path.join(dir_vectorizers, vect_files[0])
    print(f"Caricamento vectorizer: {vectorizer_path}")
    
    vectorizer = joblib.load(vectorizer_path)
    vocab = vectorizer.get_feature_names_out()
    
    # Estrai topic
    print(f"\nEstrazione top-{topn} parole per {best_k} topic...")
    
    W = lda_model.components_
    
    # Scrivi file di output (formato minimale per LLM)
    print(f"\nSalvataggio in: {output_txt}")
    
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(f"Level {level_depth} - {best_k} topics - Top {topn} keywords\n\n")
        
        for topic_idx in range(W.shape[0]):
            top_indices = np.argsort(W[topic_idx])[::-1][:topn]
            top_words = [vocab[i] for i in top_indices]
            
            f.write(f"Topic {topic_idx}: {', '.join(top_words)}\n")
    
    print(f"✓ File creato: {len(W)} topic salvati con {topn} parole ciascuno")

if __name__ == "__main__":
    main()