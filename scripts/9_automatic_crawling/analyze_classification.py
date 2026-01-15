#!/usr/bin/env python3
"""
analyze_classification.py
Analizza la classificazione dei topic e dei canali per un esperimento.

Mostra:
- Topic classificati GAMING (con keywords e messaggi rappresentativi)
- Topic classificati NON-GAMING (con keywords e messaggi rappresentativi)
- Canali TP (True Positives) con i loro messaggi
- Canali FP (False Positives) con i loro messaggi
- Canali FN (False Negatives) con i loro messaggi

METRICHE (nuova definizione):
- TP = canali con etichetta "Videogame modding" E classificati gaming
- FP = canali con etichetta ALTRO E classificati gaming
- FN = canali con etichetta "Videogame modding" E NON classificati (TUTTI, non solo visitati)

Uso:
  python analyze_classification.py --experiment threshold_40_mixed
  python analyze_classification.py --experiment threshold_15_pure --n-topics 5 --n-channels 3
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
from collections import defaultdict

# ======================== CONFIGURATION ========================
LABELED_DIR = "../../material/TGDataset/labeled_data"
RESULTS_DIR = "../../results/experiments_tgdataset"

# Target message count range for channel examples (not too few, not too many)
MIN_MESSAGES = 10
MAX_MESSAGES = 100
IDEAL_MESSAGES = 30

# ======================== HELPER FUNCTIONS ========================

def load_ground_truth():
    """Load ground truth labels."""
    gt_df = pd.read_csv(f"{LABELED_DIR}/ch_to_topic_mapping.csv")
    gaming_channels = set(gt_df[gt_df['topic'] == 'Videogame modding']['ch_ID'].tolist())
    all_labeled = set(gt_df['ch_ID'].tolist())
    non_gaming_channels = all_labeled - gaming_channels
    return gaming_channels, non_gaming_channels, all_labeled


def load_experiment_data(experiment_name, level=0):
    """Load all data for an experiment."""
    base_dir = f"{RESULTS_DIR}/{experiment_name}/level_{level}"
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Experiment not found: {base_dir}")
        return None
    
    data = {'base_dir': base_dir, 'level': level}
    
    # Load messages
    msg_file = f"{base_dir}/preprocessing/messages_english_clean.tsv.gz"
    if os.path.exists(msg_file):
        data['messages'] = pd.read_csv(msg_file, sep='\t', compression='gzip')
    else:
        print(f"[WARN] Messages file not found: {msg_file}")
        return None
    
    # Load channel stats
    stats_file = f"{base_dir}/channel_analysis/channel_stats.csv"
    if os.path.exists(stats_file):
        data['channel_stats'] = pd.read_csv(stats_file)
    
    # Load gaming topics classification
    gaming_topics_file = f"{base_dir}/classification/gaming_topics.json"
    if os.path.exists(gaming_topics_file):
        with open(gaming_topics_file) as f:
            gaming_data = json.load(f)
        data['gaming_topics'] = set(gaming_data.get('gaming_topics', []))
        data['total_topics'] = gaming_data.get('total_topics', 0)
    else:
        data['gaming_topics'] = set()
        data['total_topics'] = 0
    
    # Load gaming channels
    gaming_channels_file = f"{base_dir}/channel_analysis/gaming_channels.json"
    if os.path.exists(gaming_channels_file):
        with open(gaming_channels_file) as f:
            ch_data = json.load(f)
        data['predicted_gaming'] = set(ch_data.get('gaming_channel_ids', []))
    else:
        data['predicted_gaming'] = set()
    
    # Load LDA topic info
    lda_info_file = f"{base_dir}/lda/best_k.json"
    if os.path.exists(lda_info_file):
        with open(lda_info_file) as f:
            data['lda_info'] = json.load(f)
    
    # ============================================================
    # FIX: Load topic keywords from CORRECT path and format
    # ============================================================
    topics_file = f"{base_dir}/topics/topics_for_classification.json"
    if os.path.exists(topics_file):
        with open(topics_file) as f:
            topics_json = json.load(f)
        
        # Handle format: {"topics": [...]} or [...]
        if isinstance(topics_json, dict) and "topics" in topics_json:
            topics_list = topics_json["topics"]
        elif isinstance(topics_json, list):
            topics_list = topics_json
        else:
            topics_list = []
        
        # Convert to dict keyed by topic_id for easy lookup
        data['topics_data'] = {t['topic_id']: t for t in topics_list}
    else:
        print(f"[WARN] Topics file not found: {topics_file}")
        data['topics_data'] = {}
    
    # Load doc-topic matrix
    doc_topic_file = f"{base_dir}/lda/doc_topic_matrix_level_{level}.npy"
    if os.path.exists(doc_topic_file):
        doc_topic_matrix = np.load(doc_topic_file)
        data['dominant_topics'] = np.argmax(doc_topic_matrix, axis=1)
        data['topic_probs'] = np.max(doc_topic_matrix, axis=1)
    
    return data


def get_channel_messages(data, channel_id):
    """Get all messages for a channel."""
    df = data['messages'].copy()
    df['dominant_topic'] = data['dominant_topics']
    df['is_gaming'] = df['dominant_topic'].isin(data['gaming_topics'])
    
    ch_msgs = df[df['channel_id'] == channel_id].copy()
    return ch_msgs


def select_channels_by_message_count(channel_ids, channel_stats, n_channels=2):
    """Select channels with ideal message count (not too few, not too many)."""
    if len(channel_ids) == 0:
        return []
    
    candidates = []
    for ch_id in channel_ids:
        stats = channel_stats[channel_stats['channel_id'] == ch_id]
        if len(stats) > 0:
            msg_count = stats.iloc[0]['total_messages']
            if MIN_MESSAGES <= msg_count <= MAX_MESSAGES:
                # Score: closer to IDEAL_MESSAGES is better
                score = abs(msg_count - IDEAL_MESSAGES)
                candidates.append((ch_id, msg_count, score))
    
    # Sort by score (lower is better)
    candidates.sort(key=lambda x: x[2])
    
    # If not enough in ideal range, include some outside range
    if len(candidates) < n_channels:
        for ch_id in channel_ids:
            if ch_id not in [c[0] for c in candidates]:
                stats = channel_stats[channel_stats['channel_id'] == ch_id]
                if len(stats) > 0:
                    msg_count = stats.iloc[0]['total_messages']
                    candidates.append((ch_id, msg_count, 9999))
                    if len(candidates) >= n_channels:
                        break
    
    return candidates[:n_channels]


def print_message(msg_row, max_len=300):
    """Print a single message."""
    text = msg_row.get('text_llm') if pd.notna(msg_row.get('text_llm')) else msg_row.get('text_lda', '')
    if pd.isna(text):
        text = "[NO TEXT]"
    text = str(text)[:max_len]
    if len(str(msg_row.get('text_llm', msg_row.get('text_lda', '')))) > max_len:
        text += "..."
    return text


# ======================== MAIN ANALYSIS ========================

def analyze_topics(data, n_topics=10):
    """Analyze and display topic classification."""
    
    gaming_topics = data['gaming_topics']
    total_topics = data['total_topics']
    non_gaming_topics = set(range(total_topics)) - gaming_topics
    topics_data = data.get('topics_data', {})
    
    print(f"\n{'='*80}")
    print(f" TOPIC CLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal topics: {total_topics}")
    print(f"Gaming topics: {len(gaming_topics)} ({100*len(gaming_topics)/total_topics:.1f}%)")
    print(f"Non-gaming topics: {len(non_gaming_topics)} ({100*len(non_gaming_topics)/total_topics:.1f}%)")
    
    # ===== GAMING TOPICS =====
    print(f"\n{'='*80}")
    print(f" GAMING TOPICS (showing {min(n_topics, len(gaming_topics))} of {len(gaming_topics)})")
    print(f"{'='*80}")
    
    for i, topic_id in enumerate(list(gaming_topics)[:n_topics]):
        topic_info = topics_data.get(topic_id, {})
        keywords = topic_info.get('all_keywords', topic_info.get('keywords', []))[:15]
        top_docs = topic_info.get('top_documents', [])
        random_docs = topic_info.get('random_documents', [])
        
        print(f"\n{'─'*80}")
        print(f"TOPIC {topic_id} [GAMING]")
        print(f"{'─'*80}")
        print(f"Keywords: {', '.join(keywords) if keywords else '[not available]'}")
        
        if top_docs:
            print(f"\n  TOP 3 MESSAGGI (più rappresentativi):")
            for j, doc in enumerate(top_docs[:3]):
                text = str(doc)[:300]
                if len(str(doc)) > 300:
                    text += "..."
                print(f"    {j+1}. {text}")
        
        if random_docs:
            print(f"\n  3 MESSAGGI RANDOM:")
            for j, doc in enumerate(random_docs[:3]):
                text = str(doc)[:300]
                if len(str(doc)) > 300:
                    text += "..."
                print(f"    {j+1}. {text}")
    
    # ===== NON-GAMING TOPICS =====
    print(f"\n{'='*80}")
    print(f" NON-GAMING TOPICS (showing {min(n_topics, len(non_gaming_topics))} of {len(non_gaming_topics)})")
    print(f"{'='*80}")
    
    for i, topic_id in enumerate(list(non_gaming_topics)[:n_topics]):
        topic_info = topics_data.get(topic_id, {})
        keywords = topic_info.get('all_keywords', topic_info.get('keywords', []))[:15]
        top_docs = topic_info.get('top_documents', [])
        random_docs = topic_info.get('random_documents', [])
        
        print(f"\n{'─'*80}")
        print(f"TOPIC {topic_id} [NON-GAMING]")
        print(f"{'─'*80}")
        print(f"Keywords: {', '.join(keywords) if keywords else '[not available]'}")
        
        if top_docs:
            print(f"\n  TOP 3 MESSAGGI (più rappresentativi):")
            for j, doc in enumerate(top_docs[:3]):
                text = str(doc)[:300]
                if len(str(doc)) > 300:
                    text += "..."
                print(f"    {j+1}. {text}")
        
        if random_docs:
            print(f"\n  3 MESSAGGI RANDOM:")
            for j, doc in enumerate(random_docs[:3]):
                text = str(doc)[:300]
                if len(str(doc)) > 300:
                    text += "..."
                print(f"    {j+1}. {text}")


def analyze_channels(data, gt_gaming, gt_non_gaming, all_labeled, n_channels=2, max_messages_display=50):
    """Analyze and display channel classification."""
    
    predicted_gaming = data['predicted_gaming']
    channel_stats = data['channel_stats']
    
    # Canali con dati (presenti in channel_stats)
    channels_with_data = set(channel_stats['channel_id'].tolist())
    
    # ============================================================
    # NUOVA DEFINIZIONE METRICHE
    # ============================================================
    
    # TP: etichetta "Videogame modding" E classificato gaming
    tp_channels = predicted_gaming & gt_gaming
    
    # FP: etichetta ALTRO E classificato gaming
    fp_channels = predicted_gaming & gt_non_gaming
    
    # FN: etichetta "Videogame modding" E NON classificato
    # TUTTI i GT gaming non catturati (non solo quelli visitati)
    fn_channels_all = gt_gaming - predicted_gaming
    
    # FN con dati (per mostrare messaggi)
    fn_channels_with_data = fn_channels_all & channels_with_data
    
    # FN senza dati
    fn_channels_no_data = fn_channels_all - channels_with_data
    
    # Unlabeled
    unlabeled = predicted_gaming - all_labeled
    
    print(f"\n{'='*80}")
    print(f" CHANNEL CLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal GT gaming: {len(gt_gaming)}")
    print(f"Predicted gaming: {len(predicted_gaming)}")
    print(f"True Positives (TP): {len(tp_channels)}")
    print(f"False Positives (FP): {len(fp_channels)} (etichettati non-gaming)")
    print(f"False Negatives (FN): {len(fn_channels_all)} (GT gaming non catturati)")
    print(f"  - FN con dati: {len(fn_channels_with_data)}")
    print(f"  - FN senza dati: {len(fn_channels_no_data)}")
    print(f"Unlabeled: {len(unlabeled)} (esclusi da metriche)")
    
    # ===== TRUE POSITIVES =====
    print(f"\n{'='*80}")
    print(f" TRUE POSITIVES - Canali predetti GAMING e sono GAMING (showing {min(n_channels, len(tp_channels))})")
    print(f"{'='*80}")
    
    if len(tp_channels) == 0:
        print("\n  Nessun True Positive trovato!")
    else:
        tp_selected = select_channels_by_message_count(tp_channels, channel_stats, n_channels)
        
        for ch_id, msg_count, _ in tp_selected:
            stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
            
            print(f"\n{'─'*80}")
            print(f"CANALE {ch_id} [TRUE POSITIVE]")
            print(f"{'─'*80}")
            print(f"Ground Truth: Videogame modding (GAMING)")
            print(f"Pipeline: GAMING (ratio {stats['gaming_ratio']*100:.1f}%)")
            print(f"Messaggi totali: {int(stats['total_messages'])}")
            print(f"Messaggi gaming: {int(stats['gaming_messages'])}")
            
            ch_msgs = get_channel_messages(data, ch_id)
            
            if len(ch_msgs) > 0:
                n_show = min(len(ch_msgs), max_messages_display)
                print(f"\n  MESSAGGI ({n_show} di {len(ch_msgs)}):")
                print(f"  {'─'*70}")
                
                for j, (_, row) in enumerate(ch_msgs.head(n_show).iterrows()):
                    topic = row['dominant_topic']
                    is_gaming = "GAMING" if row['is_gaming'] else "NON-GAMING"
                    text = print_message(row, max_len=200)
                    print(f"    [{topic}][{is_gaming}] {text}")
    
    # ===== FALSE POSITIVES =====
    print(f"\n{'='*80}")
    print(f" FALSE POSITIVES - Canali predetti GAMING ma sono NON-GAMING (showing {min(n_channels, len(fp_channels))})")
    print(f"{'='*80}")
    
    if len(fp_channels) == 0:
        print("\n  Nessun False Positive trovato!")
    else:
        fp_selected = select_channels_by_message_count(fp_channels, channel_stats, n_channels)
        
        # Load GT to get actual topic
        gt_df = pd.read_csv(f"{LABELED_DIR}/ch_to_topic_mapping.csv")
        
        for ch_id, msg_count, _ in fp_selected:
            stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
            
            # Get actual GT topic
            gt_topic = gt_df[gt_df['ch_ID'] == ch_id]['topic'].values
            gt_topic = gt_topic[0] if len(gt_topic) > 0 else "Unknown"
            
            print(f"\n{'─'*80}")
            print(f"CANALE {ch_id} [FALSE POSITIVE]")
            print(f"{'─'*80}")
            print(f"Ground Truth: {gt_topic} (NON-GAMING)")
            print(f"Pipeline: GAMING (ratio {stats['gaming_ratio']*100:.1f}%) ← ERRORE")
            print(f"Messaggi totali: {int(stats['total_messages'])}")
            print(f"Messaggi gaming: {int(stats['gaming_messages'])}")
            
            ch_msgs = get_channel_messages(data, ch_id)
            
            if len(ch_msgs) > 0:
                n_show = min(len(ch_msgs), max_messages_display)
                print(f"\n  MESSAGGI ({n_show} di {len(ch_msgs)}):")
                print(f"  {'─'*70}")
                
                for j, (_, row) in enumerate(ch_msgs.head(n_show).iterrows()):
                    topic = row['dominant_topic']
                    is_gaming = "GAMING" if row['is_gaming'] else "NON-GAMING"
                    text = print_message(row, max_len=200)
                    print(f"    [{topic}][{is_gaming}] {text}")
    
    # ===== FALSE NEGATIVES (con dati) =====
    print(f"\n{'='*80}")
    print(f" FALSE NEGATIVES (con dati) - GT GAMING ma NON predetti (showing {min(n_channels, len(fn_channels_with_data))})")
    print(f"{'='*80}")
    
    if len(fn_channels_with_data) == 0:
        print("\n  Nessun False Negative con dati trovato!")
    else:
        fn_selected = select_channels_by_message_count(fn_channels_with_data, channel_stats, n_channels)
        
        for ch_id, msg_count, _ in fn_selected:
            stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
            
            print(f"\n{'─'*80}")
            print(f"CANALE {ch_id} [FALSE NEGATIVE]")
            print(f"{'─'*80}")
            print(f"Ground Truth: Videogame modding (GAMING)")
            print(f"Pipeline: NON-GAMING (ratio {stats['gaming_ratio']*100:.1f}% < threshold) ← PERSO")
            print(f"Messaggi totali: {int(stats['total_messages'])}")
            print(f"Messaggi gaming: {int(stats['gaming_messages'])}")
            
            ch_msgs = get_channel_messages(data, ch_id)
            
            if len(ch_msgs) > 0:
                n_show = min(len(ch_msgs), max_messages_display)
                print(f"\n  MESSAGGI ({n_show} di {len(ch_msgs)}):")
                print(f"  {'─'*70}")
                
                for j, (_, row) in enumerate(ch_msgs.head(n_show).iterrows()):
                    topic = row['dominant_topic']
                    is_gaming = "GAMING" if row['is_gaming'] else "NON-GAMING"
                    text = print_message(row, max_len=200)
                    print(f"    [{topic}][{is_gaming}] {text}")
    
    # ===== FALSE NEGATIVES (senza dati) =====
    if len(fn_channels_no_data) > 0:
        print(f"\n{'='*80}")
        print(f" FALSE NEGATIVES (senza dati) - GT GAMING mai visitati ({len(fn_channels_no_data)} canali)")
        print(f"{'='*80}")
        print(f"\n  Questi canali sono nel GT come 'Videogame modding' ma il crawler")
        print(f"  non li ha mai raggiunti (non sono nel dataset o non hanno messaggi inglesi).")
        print(f"\n  Primi 10 ID: {list(fn_channels_no_data)[:10]}")


def print_summary(data, gt_gaming, gt_non_gaming, all_labeled):
    """Print final summary."""
    
    predicted_gaming = data['predicted_gaming']
    gaming_topics = data['gaming_topics']
    total_topics = data['total_topics']
    
    # ============================================================
    # NUOVA DEFINIZIONE METRICHE
    # ============================================================
    
    # TP: etichetta "Videogame modding" E classificato gaming
    tp = len(predicted_gaming & gt_gaming)
    
    # FP: etichetta ALTRO E classificato gaming
    fp = len(predicted_gaming & gt_non_gaming)
    
    # FN: etichetta "Videogame modding" E NON classificato (TUTTI)
    fn = len(gt_gaming - predicted_gaming)
    
    # Precision: TP / (TP + FP)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    
    # Recall: TP / (TP + FN)
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    
    # F1
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*80}")
    print(f" SUMMARY")
    print(f"{'='*80}")
    print(f"""
┌─────────────────────────────────────────────────────────────────────────┐
│  TOPIC CLASSIFICATION                                                    │
│  ├── Total topics: {total_topics:<5}                                             │
│  ├── Gaming topics: {len(gaming_topics):<5} ({100*len(gaming_topics)/total_topics if total_topics > 0 else 0:.1f}%)                                    │
│  └── Non-gaming topics: {total_topics - len(gaming_topics):<5} ({100*(total_topics-len(gaming_topics))/total_topics if total_topics > 0 else 0:.1f}%)                                │
├─────────────────────────────────────────────────────────────────────────┤
│  CHANNEL CLASSIFICATION (nuova definizione metriche)                     │
│  ├── Total GT gaming: {len(gt_gaming):<5}                                         │
│  ├── Predicted gaming: {len(predicted_gaming):<5}                                        │
│  ├── True Positives: {tp:<5} (GT gaming + classificato gaming)            │
│  ├── False Positives: {fp:<5} (GT altro + classificato gaming)            │
│  └── False Negatives: {fn:<5} (GT gaming + NON classificato)              │
├─────────────────────────────────────────────────────────────────────────┤
│  METRICS                                                                 │
│  ├── Precision: {precision*100:.1f}% = TP/(TP+FP)                                     │
│  ├── Recall: {recall*100:.1f}% = TP/(TP+FN)                                        │
│  └── F1 Score: {f1*100:.1f}%                                                   │
└─────────────────────────────────────────────────────────────────────────┘
""")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(description='Analyze classification results for an experiment')
    parser.add_argument('--experiment', type=str, required=True,
                        help='Experiment name (e.g., threshold_40_mixed)')
    parser.add_argument('--level', type=int, default=0,
                        help='Level to analyze (default: 0)')
    parser.add_argument('--n-topics', type=int, default=10,
                        help='Number of topics to show per category (default: 10)')
    parser.add_argument('--n-channels', type=int, default=2,
                        help='Number of channels to show per category (default: 2)')
    parser.add_argument('--max-messages', type=int, default=50,
                        help='Max messages to show per channel (default: 50)')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f" CLASSIFICATION ANALYSIS: {args.experiment}")
    print(f"{'='*80}")
    print(f"Level: {args.level}")
    print(f"Topics to show: {args.n_topics} per category")
    print(f"Channels to show: {args.n_channels} per category")
    
    # Load ground truth
    gt_gaming, gt_non_gaming, all_labeled = load_ground_truth()
    print(f"\nGround Truth: {len(gt_gaming)} gaming, {len(gt_non_gaming)} non-gaming")
    
    # Load experiment data
    data = load_experiment_data(args.experiment, args.level)
    if data is None:
        return
    
    print(f"Experiment data loaded successfully")
    print(f"Topics data loaded: {len(data.get('topics_data', {}))} topics")
    
    # Analyze topics
    if data['total_topics'] > 0:
        analyze_topics(data, args.n_topics)
    else:
        print("\n[WARN] No topics found - LDA may not have run")
    
    # Analyze channels
    if len(data['predicted_gaming']) > 0 or 'channel_stats' in data:
        analyze_channels(data, gt_gaming, gt_non_gaming, all_labeled, 
                        args.n_channels, args.max_messages)
    else:
        print("\n[WARN] No channel data found")
    
    # Print summary
    print_summary(data, gt_gaming, gt_non_gaming, all_labeled)


if __name__ == "__main__":
    main()