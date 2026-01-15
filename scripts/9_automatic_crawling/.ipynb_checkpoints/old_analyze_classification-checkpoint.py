#!/usr/bin/env python3
"""
analyze_classification.py
Analizza la classificazione dei topic e dei canali per un esperimento.

Mostra:
- Topic classificati GAMING (con messaggi rappresentativi)
- Topic classificati NON-GAMING (con messaggi rappresentativi)
- Canali TP (True Positives) con i loro messaggi
- Canali FP (False Positives) con i loro messaggi
- Canali FN (False Negatives) con i loro messaggi

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
    
    # Load topic keywords
    topics_file = f"{base_dir}/lda/topics_level_{level}.json"
    if os.path.exists(topics_file):
        with open(topics_file) as f:
            data['topic_keywords'] = json.load(f)
    else:
        data['topic_keywords'] = {}
    
    # Load doc-topic matrix
    doc_topic_file = f"{base_dir}/lda/doc_topic_matrix_level_{level}.npy"
    if os.path.exists(doc_topic_file):
        doc_topic_matrix = np.load(doc_topic_file)
        data['dominant_topics'] = np.argmax(doc_topic_matrix, axis=1)
        data['topic_probs'] = np.max(doc_topic_matrix, axis=1)
    
    return data


def get_topic_messages(data, topic_id, n_top=3, n_random=3):
    """Get representative messages for a topic."""
    df = data['messages'].copy()
    df['dominant_topic'] = data['dominant_topics']
    df['topic_prob'] = data['topic_probs']
    
    topic_msgs = df[df['dominant_topic'] == topic_id].copy()
    
    if len(topic_msgs) == 0:
        return [], []
    
    # Top messages (highest probability for this topic)
    top_msgs = topic_msgs.nlargest(min(n_top, len(topic_msgs)), 'topic_prob')
    
    # Random messages
    remaining = topic_msgs[~topic_msgs.index.isin(top_msgs.index)]
    if len(remaining) > 0:
        random_msgs = remaining.sample(min(n_random, len(remaining)))
    else:
        random_msgs = pd.DataFrame()
    
    return top_msgs, random_msgs


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
        keywords = data['topic_keywords'].get(str(topic_id), [])[:15]
        
        print(f"\n{'─'*80}")
        print(f"TOPIC {topic_id} [GAMING]")
        print(f"{'─'*80}")
        print(f"Keywords: {', '.join(keywords) if keywords else '[not available]'}")
        
        top_msgs, random_msgs = get_topic_messages(data, topic_id)
        
        if len(top_msgs) > 0:
            print(f"\n  TOP 3 MESSAGGI (più rappresentativi):")
            for j, (_, row) in enumerate(top_msgs.iterrows()):
                text = print_message(row)
                print(f"    {j+1}. {text}")
        
        if len(random_msgs) > 0:
            print(f"\n  3 MESSAGGI RANDOM:")
            for j, (_, row) in enumerate(random_msgs.iterrows()):
                text = print_message(row)
                print(f"    {j+1}. {text}")
    
    # ===== NON-GAMING TOPICS =====
    print(f"\n{'='*80}")
    print(f" NON-GAMING TOPICS (showing {min(n_topics, len(non_gaming_topics))} of {len(non_gaming_topics)})")
    print(f"{'='*80}")
    
    for i, topic_id in enumerate(list(non_gaming_topics)[:n_topics]):
        keywords = data['topic_keywords'].get(str(topic_id), [])[:15]
        
        print(f"\n{'─'*80}")
        print(f"TOPIC {topic_id} [NON-GAMING]")
        print(f"{'─'*80}")
        print(f"Keywords: {', '.join(keywords) if keywords else '[not available]'}")
        
        top_msgs, random_msgs = get_topic_messages(data, topic_id)
        
        if len(top_msgs) > 0:
            print(f"\n  TOP 3 MESSAGGI (più rappresentativi):")
            for j, (_, row) in enumerate(top_msgs.iterrows()):
                text = print_message(row)
                print(f"    {j+1}. {text}")
        
        if len(random_msgs) > 0:
            print(f"\n  3 MESSAGGI RANDOM:")
            for j, (_, row) in enumerate(random_msgs.iterrows()):
                text = print_message(row)
                print(f"    {j+1}. {text}")


def analyze_channels(data, gt_gaming, gt_non_gaming, all_labeled, n_channels=2, max_messages_display=50):
    """Analyze and display channel classification."""
    
    predicted_gaming = data['predicted_gaming']
    channel_stats = data['channel_stats']
    
    # Canali con dati (presenti in channel_stats)
    channels_with_data = set(channel_stats['channel_id'].tolist())
    
    # Calculate TP, FP, FN
    tp_channels = predicted_gaming & gt_gaming
    fp_channels = predicted_gaming & gt_non_gaming  # Solo etichettati non-gaming
    fn_channels = (gt_gaming & channels_with_data) - predicted_gaming  # GT gaming con dati ma non predetti
    unlabeled = predicted_gaming - all_labeled
    
    print(f"\n{'='*80}")
    print(f" CHANNEL CLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nPredicted gaming: {len(predicted_gaming)}")
    print(f"True Positives (TP): {len(tp_channels)}")
    print(f"False Positives (FP): {len(fp_channels)} (etichettati non-gaming)")
    print(f"False Negatives (FN): {len(fn_channels)} (GT gaming ma non predetti)")
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
    
    # ===== FALSE NEGATIVES =====
    print(f"\n{'='*80}")
    print(f" FALSE NEGATIVES - Canali GT GAMING ma NON predetti (showing {min(n_channels, len(fn_channels))})")
    print(f"{'='*80}")
    
    if len(fn_channels) == 0:
        print("\n  Nessun False Negative trovato!")
    else:
        fn_selected = select_channels_by_message_count(fn_channels, channel_stats, n_channels)
        
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


def print_summary(data, gt_gaming, gt_non_gaming, all_labeled):
    """Print final summary."""
    
    predicted_gaming = data['predicted_gaming']
    gaming_topics = data['gaming_topics']
    total_topics = data['total_topics']
    
    tp = len(predicted_gaming & gt_gaming)
    fp = len(predicted_gaming & gt_non_gaming)
    
    # Get GT gaming with data
    channel_stats = data['channel_stats']
    gt_gaming_with_data = gt_gaming & set(channel_stats['channel_id'].tolist())
    fn = len(gt_gaming_with_data - predicted_gaming)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
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
│  CHANNEL CLASSIFICATION                                                  │
│  ├── Predicted gaming: {len(predicted_gaming):<5}                                        │
│  ├── True Positives: {tp:<5}                                              │
│  ├── False Positives: {fp:<5}                                             │
│  └── False Negatives: {fn:<5}                                             │
├─────────────────────────────────────────────────────────────────────────┤
│  METRICS                                                                 │
│  ├── Precision: {precision*100:.1f}%                                                  │
│  ├── Recall: {recall*100:.1f}%                                                     │
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