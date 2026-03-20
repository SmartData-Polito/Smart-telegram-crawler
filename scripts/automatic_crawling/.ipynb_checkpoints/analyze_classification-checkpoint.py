#!/usr/bin/env python3
"""
analyze_classification.py
Analizza la classificazione dei topic e dei canali per un esperimento.

METRICHE PER LIVELLO:
- GT Gaming valido in level = GT gaming valido ∩ node_ids del livello
- GT Altro valido in level = GT altro valido ∩ node_ids del livello
- TP = predicted_gaming ∩ GT gaming valido in level
- FP = predicted_gaming ∩ GT altro valido in level
- FN = GT gaming valido in level - predicted_gaming
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

Uso:
  python analyze_classification.py --experiment threshold_10_mixed_stratified
  python analyze_classification.py --experiment threshold_10_pure_stratified --level 1 --n-topics 1 --n-channels 2 --max-messages 20
"""

import os
import json
import argparse
import pandas as pd
import numpy as np
import fasttext
from collections import defaultdict
from tqdm import tqdm

# ======================== CONFIGURATION ========================
TGDATASET_DIR = "../../material"
LABELED_DIR = f"{TGDATASET_DIR}/TGDataset/labeled_data"
EXTRACTED_DIR = f"{TGDATASET_DIR}/TGDataset_extracted"
RESULTS_DIR = "../../results/experiments_tgdataset"
VALID_CHANNELS_CACHE = f"{RESULTS_DIR}/gt_valid_channels.json"
FASTTEXT_MODEL_PATH = f"{TGDATASET_DIR}/lid.176.bin"
MIN_CHANNEL_MESSAGES = 10
MIN_MESSAGE_LENGTH = 15
LANG_CONFIDENCE = 0.5

# ======================== VALID CHANNELS ========================

def load_fasttext_model():
    """Load FastText language detection model."""
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"Downloading FastText model...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
    fasttext.FastText.eprint = lambda x: None
    return fasttext.load_model(FASTTEXT_MODEL_PATH)


def count_english_messages(channel_data, ft_model):
    """Count English messages in a channel (same logic as step1_preprocess.py)."""
    text_messages = channel_data.get('text_messages', {})
    english_count = 0
    
    for msg in text_messages.values():
        msg_text = msg.get('message', '')
        if not msg_text or len(msg_text) < MIN_MESSAGE_LENGTH:
            continue
        
        text_clean = msg_text.replace('\n', ' ').strip()
        try:
            predictions = ft_model.predict(text_clean, k=1)
            lang = predictions[0][0].replace('__label__', '')
            conf = predictions[1][0]
            if lang == 'en' and conf > LANG_CONFIDENCE:
                english_count += 1
        except:
            pass
    
    return english_count


def compute_valid_channels():
    """
    Compute which GT channels have >= MIN_CHANNEL_MESSAGES ENGLISH messages.
    Uses same filtering logic as step1_preprocess.py.
    """
    print(f"\nComputing valid channels (>= {MIN_CHANNEL_MESSAGES} ENGLISH messages)...")
    print(f"This takes 2-4 hours the first time, then uses cache.")
    
    # Load GT
    topic_file = f"{LABELED_DIR}/ch_to_topic_mapping.csv"
    gt = pd.read_csv(topic_file)
    all_gt_ids = set(gt['ch_ID'].tolist())
    print(f"  Total GT channels: {len(all_gt_ids)}")
    
    # Load mapping
    mapping_file = f"{TGDATASET_DIR}/channel_file_mapping.json"
    with open(mapping_file, 'r') as f:
        mapping = json.load(f)
    
    # Load FastText
    print(f"  Loading FastText model...")
    ft_model = load_fasttext_model()
    
    # Group by file
    by_file = defaultdict(list)
    not_in_mapping = 0
    for ch_id in all_gt_ids:
        ch_str = str(ch_id)
        if ch_str in mapping:
            by_file[mapping[ch_str]['file']].append(ch_str)
        else:
            not_in_mapping += 1
    
    print(f"  Files to read: {len(by_file)}")
    print(f"  Not in mapping: {not_in_mapping}")
    
    # Count ENGLISH messages per channel
    channel_english_count = {}
    
    for file_path, ch_ids in tqdm(by_file.items(), desc="  Processing files"):
        full_path = f"{EXTRACTED_DIR}/{file_path}"
        try:
            with open(full_path) as f:
                data = json.load(f)
            for ch_str in ch_ids:
                if ch_str in data:
                    english_count = count_english_messages(data[ch_str], ft_model)
                    channel_english_count[int(ch_str)] = english_count
        except:
            pass
    
    # Channels with >= MIN_CHANNEL_MESSAGES English messages
    valid_channels = {ch for ch, count in channel_english_count.items() if count >= MIN_CHANNEL_MESSAGES}
    
    # Split by topic
    gaming_ids = set(gt[gt['topic'] == 'Videogame modding']['ch_ID'].tolist())
    other_ids = all_gt_ids - gaming_ids
    
    gaming_valid = gaming_ids & valid_channels
    other_valid = other_ids & valid_channels
    
    # Save cache
    os.makedirs(os.path.dirname(VALID_CHANNELS_CACHE), exist_ok=True)
    result = {
        'min_messages': MIN_CHANNEL_MESSAGES,
        'total_gt': len(all_gt_ids),
        'gaming': {
            'total': len(gaming_ids),
            'valid': len(gaming_valid),
            'valid_ids': list(gaming_valid),
        },
        'other': {
            'total': len(other_ids),
            'valid': len(other_valid),
            'valid_ids': list(other_valid),
        },
    }
    
    with open(VALID_CHANNELS_CACHE, 'w') as f:
        json.dump(result, f)
    
    print(f"  Gaming valid: {len(gaming_valid)} / {len(gaming_ids)} ({100*len(gaming_valid)/len(gaming_ids):.1f}%)")
    print(f"  Other valid: {len(other_valid)} / {len(other_ids)} ({100*len(other_valid)/len(other_ids):.1f}%)")
    print(f"  Saved cache: {VALID_CHANNELS_CACHE}")
    
    return gaming_valid, other_valid


def load_valid_channels():
    """Load or compute valid channels."""
    if os.path.exists(VALID_CHANNELS_CACHE):
        with open(VALID_CHANNELS_CACHE, 'r') as f:
            data = json.load(f)
        
        if data.get('min_messages') == MIN_CHANNEL_MESSAGES:
            gaming_valid = set(data['gaming']['valid_ids'])
            other_valid = set(data['other']['valid_ids'])
            print(f"\nLoaded valid channels from cache:")
            print(f"  Gaming valid: {len(gaming_valid)}")
            print(f"  Other valid: {len(other_valid)}")
            return gaming_valid, other_valid
    
    return compute_valid_channels()


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
    topics_file = f"{base_dir}/topics/topics_for_classification.json"
    if os.path.exists(topics_file):
        with open(topics_file) as f:
            topics_json = json.load(f)
        
        if isinstance(topics_json, dict) and "topics" in topics_json:
            topics_list = topics_json["topics"]
        elif isinstance(topics_json, list):
            topics_list = topics_json
        else:
            topics_list = []
        
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


def select_channels(channel_ids, channel_stats, n_channels=2):
    """Select n_channels from channel_ids."""
    if len(channel_ids) == 0:
        return []
    
    candidates = []
    for ch_id in channel_ids:
        stats = channel_stats[channel_stats['channel_id'] == ch_id]
        if len(stats) > 0:
            msg_count = stats.iloc[0]['total_messages']
            candidates.append((ch_id, msg_count))
    
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
    
    # GAMING TOPICS
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
    
    # NON-GAMING TOPICS
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


def analyze_channels(data, gt_gaming_valid, gt_non_gaming_valid, all_labeled, n_channels=2, max_messages_display=50):
    """Analyze and display channel classification."""
    
    predicted_gaming = data['predicted_gaming']
    channel_stats = data['channel_stats']
    
    channels_with_data = set(channel_stats['channel_id'].tolist())
    
    # METRICHE
    tp_channels = predicted_gaming & gt_gaming_valid
    fp_channels = predicted_gaming & gt_non_gaming_valid
    fn_channels_all = gt_gaming_valid - predicted_gaming
    fn_channels_with_data = fn_channels_all & channels_with_data
    fn_channels_no_data = fn_channels_all - channels_with_data
    unlabeled = predicted_gaming - all_labeled
    
    print(f"\n{'='*80}")
    print(f" CHANNEL CLASSIFICATION ANALYSIS")
    print(f"{'='*80}")
    print(f"\nGT gaming validi: {len(gt_gaming_valid)}")
    print(f"GT non-gaming validi: {len(gt_non_gaming_valid)}")
    print(f"Predicted gaming: {len(predicted_gaming)}")
    print(f"True Positives (TP): {len(tp_channels)}")
    print(f"False Positives (FP): {len(fp_channels)}")
    print(f"False Negatives (FN): {len(fn_channels_all)}")
    print(f"  - FN con dati: {len(fn_channels_with_data)}")
    print(f"  - FN senza dati: {len(fn_channels_no_data)}")
    print(f"Unlabeled: {len(unlabeled)} (esclusi da metriche)")
    
    # TRUE POSITIVES
    print(f"\n{'='*80}")
    print(f" TRUE POSITIVES (showing {min(n_channels, len(tp_channels))})")
    print(f"{'='*80}")
    
    if len(tp_channels) == 0:
        print("\n  Nessun True Positive trovato!")
    else:
        tp_selected = select_channels(tp_channels, channel_stats, n_channels)
        
        for ch_id, msg_count in tp_selected:
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
    
    # FALSE POSITIVES
    print(f"\n{'='*80}")
    print(f" FALSE POSITIVES (showing {min(n_channels, len(fp_channels))})")
    print(f"{'='*80}")
    
    if len(fp_channels) == 0:
        print("\n  Nessun False Positive trovato!")
    else:
        fp_selected = select_channels(fp_channels, channel_stats, n_channels)
        gt_df = pd.read_csv(f"{LABELED_DIR}/ch_to_topic_mapping.csv")
        
        for ch_id, msg_count in fp_selected:
            stats = channel_stats[channel_stats['channel_id'] == ch_id].iloc[0]
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
    
    # FALSE NEGATIVES (con dati)
    print(f"\n{'='*80}")
    print(f" FALSE NEGATIVES con dati (showing {min(n_channels, len(fn_channels_with_data))})")
    print(f"{'='*80}")
    
    if len(fn_channels_with_data) == 0:
        print("\n  Nessun False Negative con dati trovato!")
    else:
        fn_selected = select_channels(fn_channels_with_data, channel_stats, n_channels)
        
        for ch_id, msg_count in fn_selected:
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
    
    # FALSE NEGATIVES (senza dati)
    if len(fn_channels_no_data) > 0:
        print(f"\n{'='*80}")
        print(f" FALSE NEGATIVES senza dati ({len(fn_channels_no_data)} canali)")
        print(f"{'='*80}")
        print(f"\n  Questi canali sono GT gaming validi ma il crawler non li ha raggiunti.")
        print(f"\n  Primi 10 ID: {list(fn_channels_no_data)[:10]}")


def print_summary(data, gt_gaming_valid, gt_non_gaming_valid, all_labeled, gt_gaming_total):
    """Print final summary."""
    
    predicted_gaming = data['predicted_gaming']
    gaming_topics = data['gaming_topics']
    total_topics = data['total_topics']
    
    tp = len(predicted_gaming & gt_gaming_valid)
    fp = len(predicted_gaming & gt_non_gaming_valid)
    fn = len(gt_gaming_valid - predicted_gaming)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"\n{'='*80}")
    print(f" SUMMARY")
    print(f"{'='*80}")
    print(f"""
  TOPIC CLASSIFICATION
  ├── Total topics: {total_topics}
  ├── Gaming topics: {len(gaming_topics)} ({100*len(gaming_topics)/total_topics if total_topics > 0 else 0:.1f}%)
  └── Non-gaming topics: {total_topics - len(gaming_topics)}

  GROUND TRUTH
  ├── GT gaming totali: {gt_gaming_total}
  ├── GT gaming validi: {len(gt_gaming_valid)} ({100*len(gt_gaming_valid)/gt_gaming_total:.1f}%)
  └── GT non-gaming validi: {len(gt_non_gaming_valid)}

  CHANNEL CLASSIFICATION
  ├── Predicted gaming: {len(predicted_gaming)}
  ├── True Positives: {tp}
  ├── False Positives: {fp}
  └── False Negatives: {fn}

  METRICS
  ├── Precision: {precision*100:.1f}%
  ├── Recall: {recall*100:.1f}%
  └── F1 Score: {f1*100:.1f}%
""")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--level', type=int, default=0)
    parser.add_argument('--n-topics', type=int, default=10)
    parser.add_argument('--n-channels', type=int, default=2)
    parser.add_argument('--max-messages', type=int, default=50)
    parser.add_argument('--recompute-valid', action='store_true')
    
    args = parser.parse_args()
    
    print(f"\n{'='*80}")
    print(f" CLASSIFICATION ANALYSIS: {args.experiment}")
    print(f"{'='*80}")
    print(f"Level: {args.level}")
    
    if args.recompute_valid and os.path.exists(VALID_CHANNELS_CACHE):
        os.remove(VALID_CHANNELS_CACHE)
        print("Removed valid channels cache.")
    
    # Load ground truth
    gt_gaming, gt_non_gaming, all_labeled = load_ground_truth()
    print(f"\nGround Truth (tutti): {len(gt_gaming)} gaming, {len(gt_non_gaming)} non-gaming")
    
    # Load valid channels
    gt_gaming_valid, gt_non_gaming_valid = load_valid_channels()
    
    # Load experiment data
    data = load_experiment_data(args.experiment, args.level)
    if data is None:
        return
    
    print(f"Experiment data loaded successfully")
    
    # Analyze topics
    if data['total_topics'] > 0:
        analyze_topics(data, args.n_topics)
    
    # Analyze channels
    if len(data['predicted_gaming']) > 0 or 'channel_stats' in data:
        analyze_channels(data, gt_gaming_valid, gt_non_gaming_valid, all_labeled, 
                        args.n_channels, args.max_messages)
    
    # Print summary
    print_summary(data, gt_gaming_valid, gt_non_gaming_valid, all_labeled, len(gt_gaming))


if __name__ == "__main__":
    main()