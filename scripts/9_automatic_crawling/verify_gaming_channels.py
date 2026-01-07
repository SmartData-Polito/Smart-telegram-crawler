#!/usr/bin/env python3
"""
verify_gaming_channels.py
Mostra messaggi reali dei canali classificati come gaming per verifica manuale.
"""

import json
import pandas as pd
import random

# Config
EXPERIMENT = "threshold_40_pure"
LEVEL = 0
N_CHANNELS = 5  # Quanti canali mostrare
N_MESSAGES = 10  # Quanti messaggi per canale

BASE_DIR = f"../../results/experiments_tgdataset/{EXPERIMENT}/level_{LEVEL}"

def main():
    # 1. Carica canali gaming trovati dalla pipeline
    with open(f"{BASE_DIR}/channel_analysis/gaming_channels.json") as f:
        gaming_data = json.load(f)
    
    gaming_channel_ids = gaming_data['gaming_channel_ids']
    print(f"Pipeline ha trovato {len(gaming_channel_ids)} canali gaming")
    print(f"Threshold: {gaming_data['threshold']*100:.0f}%")
    print(f"Mean gaming ratio: {gaming_data['mean_gaming_ratio']*100:.1f}%")
    
    # 2. Carica messaggi
    df = pd.read_csv(f"{BASE_DIR}/preprocessing/messages_english_clean.tsv.gz", 
                     sep='\t', compression='gzip')
    
    # 3. Carica gaming topics
    with open(f"{BASE_DIR}/classification/gaming_topics.json") as f:
        gaming_topics = set(json.load(f)['gaming_topics'])
    
    print(f"\nGaming topic IDs: {sorted(gaming_topics)}")
    
    # 4. Carica doc-topic matrix per vedere topic dominante
    import numpy as np
    doc_topic_matrix = np.load(f"{BASE_DIR}/lda/doc_topic_matrix_level_{LEVEL}.npy")
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    df['dominant_topic'] = dominant_topics
    df['is_gaming_msg'] = df['dominant_topic'].isin(gaming_topics)
    
    # 5. Carica channel stats
    channel_stats = pd.read_csv(f"{BASE_DIR}/channel_analysis/channel_stats.csv")
    
    # 6. Seleziona canali random da mostrare
    sample_channels = random.sample(gaming_channel_ids, min(N_CHANNELS, len(gaming_channel_ids)))
    
    print("\n" + "="*80)
    print(" VERIFICA MANUALE CANALI GAMING")
    print("="*80)
    
    for ch_id in sample_channels:
        ch_stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
        ch_messages = df[df['channel_id'] == ch_id]
        
        print(f"\n{'='*80}")
        print(f"CANALE: {ch_id}")
        print(f"{'='*80}")
        print(f"  Messaggi totali: {ch_stats['total_messages']}")
        print(f"  Messaggi gaming: {ch_stats['gaming_messages']}")
        print(f"  Gaming ratio: {ch_stats['gaming_ratio']*100:.1f}%")
        
        # Mostra messaggi gaming
        gaming_msgs = ch_messages[ch_messages['is_gaming_msg'] == True]
        non_gaming_msgs = ch_messages[ch_messages['is_gaming_msg'] == False]
        
        print(f"\n--- MESSAGGI GAMING (topic in {gaming_topics}) ---")
        sample_gaming = gaming_msgs.sample(min(N_MESSAGES//2, len(gaming_msgs))) if len(gaming_msgs) > 0 else gaming_msgs
        for _, row in sample_gaming.iterrows():
            text = row['text_llm'][:200] if pd.notna(row['text_llm']) else row['text_lda'][:200]
            print(f"  [Topic {row['dominant_topic']}] {text}...")
        
        print(f"\n--- MESSAGGI NON-GAMING ---")
        sample_non = non_gaming_msgs.sample(min(N_MESSAGES//2, len(non_gaming_msgs))) if len(non_gaming_msgs) > 0 else non_gaming_msgs
        for _, row in sample_non.iterrows():
            text = row['text_llm'][:200] if pd.notna(row['text_llm']) else row['text_lda'][:200]
            print(f"  [Topic {row['dominant_topic']}] {text}...")
        
        print(f"\n>>> DOMANDA: Classificheresti questo canale come GAMING? (ratio {ch_stats['gaming_ratio']*100:.1f}%)")
    
    # 7. Mostra anche canali NON classificati come gaming ma che sono nella ground truth
    print("\n\n" + "="*80)
    print(" CANALI GAMING (ground truth) NON RICONOSCIUTI DALLA PIPELINE")
    print("="*80)
    
    # Tutti i canali nel level 0 sono gaming (ground truth per pure experiment)
    all_channel_ids = df['channel_id'].unique().tolist()
    missed_channels = [ch for ch in all_channel_ids if ch not in gaming_channel_ids]
    
    # Mostra qualche esempio di canali persi
    sample_missed = random.sample(missed_channels, min(3, len(missed_channels)))
    
    for ch_id in sample_missed:
        ch_stats_row = channel_stats[channel_stats['channel_id'] == ch_id]
        if len(ch_stats_row) == 0:
            continue
        ch_stats = ch_stats_row.iloc[0]
        ch_messages = df[df['channel_id'] == ch_id]
        
        print(f"\n{'='*80}")
        print(f"CANALE PERSO: {ch_id} (è gaming nella ground truth ma NON riconosciuto)")
        print(f"{'='*80}")
        print(f"  Gaming ratio: {ch_stats['gaming_ratio']*100:.1f}% (sotto threshold 40%)")
        
        sample_msgs = ch_messages.sample(min(5, len(ch_messages)))
        for _, row in sample_msgs.iterrows():
            text = row['text_llm'][:200] if pd.notna(row['text_llm']) else row['text_lda'][:200]
            print(f"  [Topic {row['dominant_topic']}] {text}...")
        
        print(f"\n>>> DOMANDA: Questo canale parla di gaming? (ratio solo {ch_stats['gaming_ratio']*100:.1f}%)")

if __name__ == "__main__":
    main()