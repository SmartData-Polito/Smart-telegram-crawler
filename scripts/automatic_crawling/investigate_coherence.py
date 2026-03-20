#!/usr/bin/env python3
"""
investigate_coherence.py
Analizza la coherence per singolo topic per capire quali sono problematici.
"""

import json
import joblib
import numpy as np
from gensim.models import CoherenceModel
from gensim.corpora import Dictionary
import pandas as pd
import os

# Config
EXPERIMENT = "threshold_20_pure"
LEVEL = 0

BASE_DIR = f"../../results/experiments_tgdataset/{EXPERIMENT}/level_{LEVEL}"

def main():
    print(f"Analyzing coherence for {EXPERIMENT} level {LEVEL}")
    
    # 0. Verifica file disponibili
    print("\nChecking available files...")
    lda_dir = f"{BASE_DIR}/lda"
    models_dir = f"{BASE_DIR}/lda/models"
    
    if os.path.exists(lda_dir):
        print(f"  {lda_dir}: {os.listdir(lda_dir)}")
    if os.path.exists(models_dir):
        print(f"  {models_dir}: {os.listdir(models_dir)}")
    
    # 1. Carica modello LDA
    print("\nLoading LDA model...")
    lda_model = joblib.load(f"{models_dir}/lda_best.joblib")
    print(f"  Num topics: {lda_model.num_topics}")
    
    # 2. Carica testi
    print("Loading texts...")
    df = pd.read_csv(f"{BASE_DIR}/preprocessing/messages_english_clean.tsv.gz",
                     sep='\t', compression='gzip')
    texts = [str(doc).split() for doc in df['text_lda'].fillna('')]
    print(f"  Documents: {len(texts)}")
    
    # 3. Ricostruisci dictionary dai testi
    print("\nBuilding dictionary from texts...")
    dictionary = Dictionary(texts)
    print(f"  Vocab size: {len(dictionary)}")
    
    # 4. Calcola coherence per topic
    print(f"\nCalculating coherence for {lda_model.num_topics} topics...")
    print("  (This may take a few minutes...)")
    
    coherence_model = CoherenceModel(
        model=lda_model,
        texts=texts,
        dictionary=dictionary,
        coherence='c_npmi'
    )
    
    # Coherence per singolo topic
    coherence_per_topic = coherence_model.get_coherence_per_topic()
    
    # 5. Analisi
    print(f"\n{'='*60}")
    print(f" COHERENCE PER TOPIC")
    print(f"{'='*60}")
    
    # Conta problematici
    n_inf = sum(1 for c in coherence_per_topic if np.isinf(c))
    n_nan = sum(1 for c in coherence_per_topic if np.isnan(c))
    n_valid = sum(1 for c in coherence_per_topic if np.isfinite(c))
    
    print(f"\nTotale topic: {len(coherence_per_topic)}")
    print(f"  Validi (finiti): {n_valid}")
    print(f"  Infiniti: {n_inf}")
    print(f"  NaN: {n_nan}")
    
    # Statistiche sui validi
    valid_coherences = [c for c in coherence_per_topic if np.isfinite(c)]
    if valid_coherences:
        print(f"\nStatistiche sui {n_valid} topic validi:")
        print(f"  Min:    {min(valid_coherences):.4f}")
        print(f"  Max:    {max(valid_coherences):.4f}")
        print(f"  Mean:   {np.mean(valid_coherences):.4f}")
        print(f"  Median: {np.median(valid_coherences):.4f}")
    
    # Mostra topic problematici
    print(f"\n{'='*60}")
    print(f" TOPIC PROBLEMATICI (infinito/NaN)")
    print(f"{'='*60}")
    
    problematic = [(i, c) for i, c in enumerate(coherence_per_topic) 
                   if not np.isfinite(c)]
    
    if problematic:
        print(f"\n{len(problematic)} topic con coherence infinita/NaN:")
        for topic_id, coh in problematic[:20]:  # Mostra primi 20
            # Mostra top words del topic
            top_words = lda_model.show_topic(topic_id, topn=10)
            words = [w for w, _ in top_words]
            print(f"  Topic {topic_id}: {coh} - {', '.join(words[:5])}...")
    else:
        print("\nNessun topic problematico!")
    
    # Mostra distribuzione
    print(f"\n{'='*60}")
    print(f" DISTRIBUZIONE COHERENCE")
    print(f"{'='*60}")
    
    if valid_coherences:
        bins = [(-1, -0.2), (-0.2, -0.1), (-0.1, 0), (0, 0.1), (0.1, 1)]
        for low, high in bins:
            count = sum(1 for c in valid_coherences if low <= c < high)
            pct = 100 * count / len(valid_coherences) if valid_coherences else 0
            print(f"  [{low:+.1f}, {high:+.1f}): {count:4d} topic ({pct:.1f}%)")
    
    # Mostra top e bottom topic
    print(f"\n{'='*60}")
    print(f" TOP 10 TOPIC (coherence più alta)")
    print(f"{'='*60}")
    
    sorted_topics = sorted(enumerate(coherence_per_topic), 
                          key=lambda x: x[1] if np.isfinite(x[1]) else -999)
    
    for topic_id, coh in sorted_topics[-10:][::-1]:
        if np.isfinite(coh):
            top_words = lda_model.show_topic(topic_id, topn=5)
            words = [w for w, _ in top_words]
            print(f"  Topic {topic_id}: {coh:+.4f} - {', '.join(words)}")
    
    print(f"\n{'='*60}")
    print(f" BOTTOM 10 TOPIC (coherence più bassa, esclusi infiniti)")
    print(f"{'='*60}")
    
    for topic_id, coh in sorted_topics[:10]:
        if np.isfinite(coh):
            top_words = lda_model.show_topic(topic_id, topn=5)
            words = [w for w, _ in top_words]
            print(f"  Topic {topic_id}: {coh:+.4f} - {', '.join(words)}")
    
    # Salva risultati
    results = {
        'experiment': EXPERIMENT,
        'level': LEVEL,
        'total_topics': len(coherence_per_topic),
        'valid_topics': n_valid,
        'infinite_topics': n_inf,
        'nan_topics': n_nan,
        'mean_valid': float(np.mean(valid_coherences)) if valid_coherences else None,
        'median_valid': float(np.median(valid_coherences)) if valid_coherences else None,
        'coherence_per_topic': [float(c) if np.isfinite(c) else str(c) for c in coherence_per_topic]
    }
    
    output_file = f"{BASE_DIR}/lda/coherence_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {output_file}")

if __name__ == "__main__":
    main()