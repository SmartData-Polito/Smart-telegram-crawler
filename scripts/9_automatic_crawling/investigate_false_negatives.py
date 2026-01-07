#!/usr/bin/env python3
"""
investigate_false_negatives.py
Investiga i False Negatives - canali gaming (GT) che la pipeline NON ha riconosciuto.
ESCLUSI: canali senza dati (no messaggi inglesi)
"""

import json
import pandas as pd
import numpy as np
import random

# Config
EXPERIMENT = "threshold_20_pure"
LEVEL = 0
N_EXAMPLES = 10

BASE_DIR = f"../../results/experiments_tgdataset/{EXPERIMENT}/level_{LEVEL}"
LABELED_DIR = "../../material/TGDataset/labeled_data"

def main():
    # 1. Carica ground truth gaming
    gt_df = pd.read_csv(f"{LABELED_DIR}/ch_to_topic_mapping.csv")
    gt_gaming = set(gt_df[gt_df['topic'] == 'Videogame modding']['ch_ID'].tolist())
    print(f"Ground truth gaming: {len(gt_gaming)} canali")
    
    # 2. Carica canali classificati gaming dalla pipeline
    with open(f"{BASE_DIR}/channel_analysis/gaming_channels.json") as f:
        gaming_data = json.load(f)
    
    predicted_gaming = set(gaming_data['gaming_channel_ids'])
    print(f"Pipeline predicted gaming: {len(predicted_gaming)} canali")
    
    # 3. Calcola False Negatives
    true_positives = predicted_gaming & gt_gaming
    false_negatives = gt_gaming - predicted_gaming
    
    print(f"\nTrue Positives: {len(true_positives)}")
    print(f"False Negatives totali: {len(false_negatives)}")
    
    # 4. Carica messaggi e stats
    df = pd.read_csv(f"{BASE_DIR}/preprocessing/messages_english_clean.tsv.gz", 
                     sep='\t', compression='gzip')
    
    channel_stats = pd.read_csv(f"{BASE_DIR}/channel_analysis/channel_stats.csv")
    
    # 5. Carica gaming topics
    with open(f"{BASE_DIR}/classification/gaming_topics.json") as f:
        gaming_topics = set(json.load(f)['gaming_topics'])
    
    # 6. Carica doc-topic matrix
    doc_topic_matrix = np.load(f"{BASE_DIR}/lda/doc_topic_matrix_level_{LEVEL}.npy")
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    df['dominant_topic'] = dominant_topics
    df['is_gaming_msg'] = df['dominant_topic'].isin(gaming_topics)
    
    # 7. Separa FN con dati e senza dati
    fn_with_stats = []
    for ch_id in false_negatives:
        ch_stats = channel_stats[channel_stats['channel_id'] == ch_id]
        if len(ch_stats) > 0:
            fn_with_stats.append({
                'channel_id': ch_id,
                'gaming_ratio': ch_stats.iloc[0]['gaming_ratio'],
                'total_messages': ch_stats.iloc[0]['total_messages'],
                'gaming_messages': ch_stats.iloc[0]['gaming_messages']
            })
    
    fn_df = pd.DataFrame(fn_with_stats)
    
    # Canali FN senza dati (non analizzabili)
    fn_no_data = len(false_negatives) - len(fn_df)
    fn_analyzable = len(fn_df)
    
    print(f"\n" + "="*70)
    print(f" FALSE NEGATIVES: DATI vs NON ANALIZZABILI")
    print(f"="*70)
    print(f"\nFalse Negatives totali: {len(false_negatives)}")
    print(f"  - CON dati (analizzabili): {fn_analyzable}")
    print(f"  - SENZA dati (esclusi):    {fn_no_data}")
    
    # 8. Analizza SOLO i canali con dati
    print(f"\n" + "="*70)
    print(f" DISTRIBUZIONE GAMING RATIO (solo {fn_analyzable} analizzabili)")
    print(f"="*70)
    
    if len(fn_df) > 0:
        print(f"\nStatistiche gaming_ratio:")
        print(f"  Min:    {fn_df['gaming_ratio'].min()*100:.1f}%")
        print(f"  Max:    {fn_df['gaming_ratio'].max()*100:.1f}%")
        print(f"  Mean:   {fn_df['gaming_ratio'].mean()*100:.1f}%")
        print(f"  Median: {fn_df['gaming_ratio'].median()*100:.1f}%")
        
        # Distribuzione per fasce (percentuali su fn_analyzable, NON su totale)
        print(f"\nDistribuzione per fascia (su {fn_analyzable} canali analizzabili):")
        
        # Conta per fascia
        very_low = fn_df[fn_df['gaming_ratio'] < 0.05]
        low = fn_df[(fn_df['gaming_ratio'] >= 0.05) & (fn_df['gaming_ratio'] < 0.10)]
        medium = fn_df[(fn_df['gaming_ratio'] >= 0.10) & (fn_df['gaming_ratio'] < 0.15)]
        near_threshold = fn_df[(fn_df['gaming_ratio'] >= 0.15) & (fn_df['gaming_ratio'] < 0.20)]
        
        print(f"  0-5%   (GT probabilmente errata):  {len(very_low):4d} ({100*len(very_low)/fn_analyzable:.1f}%)")
        print(f"  5-10%  (contenuto molto misto):    {len(low):4d} ({100*len(low)/fn_analyzable:.1f}%)")
        print(f"  10-15% (contenuto misto):          {len(medium):4d} ({100*len(medium)/fn_analyzable:.1f}%)")
        print(f"  15-20% (vicini a threshold):       {len(near_threshold):4d} ({100*len(near_threshold)/fn_analyzable:.1f}%)")
        
        # Quanti hanno 0 messaggi gaming?
        zero_gaming = len(fn_df[fn_df['gaming_messages'] == 0])
        print(f"\n  Con 0 messaggi gaming: {zero_gaming} ({100*zero_gaming/fn_analyzable:.1f}%)")
    
    # 9. Mostra esempi di FN per ogni categoria
    print(f"\n" + "="*70)
    print(f" ESEMPI DI FALSE NEGATIVES PER CATEGORIA")
    print(f"="*70)
    
    categories = [
        ("VICINI A THRESHOLD (15-20%)", fn_df[fn_df['gaming_ratio'] >= 0.15], "Vero gaming, perso per poco"),
        ("CONTENUTO MISTO (10-15%)", fn_df[(fn_df['gaming_ratio'] >= 0.10) & (fn_df['gaming_ratio'] < 0.15)], "Gaming + altro"),
        ("MOLTO MISTO (5-10%)", fn_df[(fn_df['gaming_ratio'] >= 0.05) & (fn_df['gaming_ratio'] < 0.10)], "Gaming minoritario"),
        ("RATIO BASSO (0-5%)", fn_df[fn_df['gaming_ratio'] < 0.05], "Probabilmente GT errata"),
    ]
    
    examples_shown = 0
    for cat_name, cat_df, interpretation in categories:
        if len(cat_df) == 0 or examples_shown >= N_EXAMPLES:
            continue
            
        # Prendi 2-3 esempi per categoria
        n_samples = min(2, len(cat_df), N_EXAMPLES - examples_shown)
        samples = cat_df.sample(n_samples)['channel_id'].tolist()
        
        for ch_id in samples:
            ch_stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
            ch_messages = df[df['channel_id'] == ch_id]
            
            print(f"\n{'='*70}")
            print(f"[{cat_name}] CANALE: {ch_id}")
            print(f"{'='*70}")
            print(f"  Ground Truth: 'Videogame modding' (GAMING)")
            print(f"  Pipeline: NON gaming (ratio {ch_stats['gaming_ratio']*100:.1f}% < 20%)")
            print(f"  Messaggi totali: {int(ch_stats['total_messages'])}")
            print(f"  Messaggi gaming: {int(ch_stats['gaming_messages'])}")
            print(f"  Interpretazione: {interpretation}")
            
            if len(ch_messages) > 0:
                gaming_msgs = ch_messages[ch_messages['is_gaming_msg'] == True]
                non_gaming_msgs = ch_messages[ch_messages['is_gaming_msg'] == False]
                
                print(f"\n--- MESSAGGI GAMING ({len(gaming_msgs)}) ---")
                for _, row in gaming_msgs.head(2).iterrows():
                    text = row['text_llm'][:200] if pd.notna(row['text_llm']) else row['text_lda'][:200]
                    print(f"  [Topic {row['dominant_topic']}] {text}...")
                
                print(f"\n--- MESSAGGI NON-GAMING ({len(non_gaming_msgs)}) ---")
                for _, row in non_gaming_msgs.head(2).iterrows():
                    text = row['text_llm'][:200] if pd.notna(row['text_llm']) else row['text_lda'][:200]
                    print(f"  [Topic {row['dominant_topic']}] {text}...")
            
            examples_shown += 1
    
    # 10. Riepilogo finale (SOLO su analizzabili)
    print(f"\n\n" + "="*70)
    print(f" RIEPILOGO FALSE NEGATIVES (SOLO ANALIZZABILI)")
    print(f"="*70)
    
    print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  FALSE NEGATIVES TOTALI: {len(false_negatives):4d}                                   │
│  ├── Analizzabili (con dati): {fn_analyzable:4d}                              │
│  └── Non analizzabili (senza dati): {fn_no_data:4d} (ESCLUSI)                 │
└─────────────────────────────────────────────────────────────────┘

Distribuzione dei {fn_analyzable} FN ANALIZZABILI:

  1. RATIO 0-5% (GT probabilmente errata):   {len(very_low):4d} ({100*len(very_low)/fn_analyzable:.1f}%)
     → Pipeline CORRETTA a escluderli
     
  2. RATIO 5-10% (contenuto molto misto):    {len(low):4d} ({100*len(low)/fn_analyzable:.1f}%)
     → Gaming è attività minoritaria
     
  3. RATIO 10-15% (contenuto misto):         {len(medium):4d} ({100*len(medium)/fn_analyzable:.1f}%)
     → Trade-off threshold
     
  4. RATIO 15-20% (vicini a threshold):      {len(near_threshold):4d} ({100*len(near_threshold)/fn_analyzable:.1f}%)
     → Veri gaming persi per poco

──────────────────────────────────────────────────────────────────
INTERPRETAZIONE:

  ✓ Pipeline CORRETTA:  {len(very_low):4d} ({100*len(very_low)/fn_analyzable:.1f}%) - GT errata
  ~ Trade-off:          {len(low)+len(medium):4d} ({100*(len(low)+len(medium))/fn_analyzable:.1f}%) - Contenuto misto
  ✗ Pipeline SBAGLIA:   {len(near_threshold):4d} ({100*len(near_threshold)/fn_analyzable:.1f}%) - Veri gaming persi

──────────────────────────────────────────────────────────────────
RECALL CALCULATIONS:

  Recall originale:     {len(true_positives)}/{len(gt_gaming)} = {100*len(true_positives)/len(gt_gaming):.1f}%
  
  Escludendo non analizzabili ({fn_no_data}):
  Recall adjusted:      {len(true_positives)}/{len(gt_gaming)-fn_no_data} = {100*len(true_positives)/(len(gt_gaming)-fn_no_data):.1f}%
  
  Escludendo anche GT errata ({len(very_low)}):
  Recall "vera":        {len(true_positives)}/{len(gt_gaming)-fn_no_data-len(very_low)} = {100*len(true_positives)/(len(gt_gaming)-fn_no_data-len(very_low)):.1f}%
""")

if __name__ == "__main__":
    main()