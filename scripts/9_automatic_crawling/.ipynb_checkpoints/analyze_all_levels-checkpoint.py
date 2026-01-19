#!/usr/bin/env python3
"""
Analyze results from TGDataset gaming detection pipeline.

CANALI VALIDI:
- Solo canali con >= 10 messaggi INGLESI nel TGDataset
- Canali con < 10 messaggi inglesi sono ESCLUSI da tutte le metriche

METRICHE PER LIVELLO:
- GT Gaming valido in level = GT gaming valido ∩ node_ids del livello
- GT Altro valido in level = GT altro valido ∩ node_ids del livello
- TP = predicted_gaming ∩ GT gaming valido in level
- FP = predicted_gaming ∩ GT altro valido in level
- FN = GT gaming valido in level - predicted_gaming
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

METRICHE GLOBALI:
- GT Gaming valido = canali "Videogame modding" con >= 10 msg inglesi
- GT Altro valido = canali con altra etichetta con >= 10 msg inglesi
- TP = predicted_gaming ∩ GT gaming valido
- FP = predicted_gaming ∩ GT altro valido
- FN = GT gaming valido - TP
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
"""

import os
import json
import argparse
import pandas as pd
import fasttext
from collections import defaultdict
from tqdm import tqdm

# ======================== CONFIGURATION ========================
TGDATASET_DIR = "../../material"
LABELED_DATA_DIR = f"{TGDATASET_DIR}/TGDataset/labeled_data"
EXTRACTED_DIR = f"{TGDATASET_DIR}/TGDataset_extracted"
VALID_CHANNELS_CACHE = "../../results/experiments_tgdataset/gt_valid_channels.json"
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
    topic_file = f"{LABELED_DATA_DIR}/ch_to_topic_mapping.csv"
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


# ======================== GROUND TRUTH ========================

def load_ground_truth():
    """Load ground truth labels."""
    topic_file = f"{LABELED_DATA_DIR}/ch_to_topic_mapping.csv"
    
    if not os.path.exists(topic_file):
        return None, set(), set(), set()
    
    df = pd.read_csv(topic_file)
    gaming_channels = set(df[df['topic'] == 'Videogame modding']['ch_ID'].tolist())
    all_labeled_channels = set(df['ch_ID'].tolist())
    non_gaming_labeled = all_labeled_channels - gaming_channels
    
    return df, gaming_channels, non_gaming_labeled, all_labeled_channels


# ======================== LEVEL ANALYSIS ========================

def analyze_level(level, base_dir, threshold, gt_gaming_valid, gt_non_gaming_valid, all_labeled_channels):
    """Analyze a single level."""
    
    level_dir = f"{base_dir}/level_{level}"
    
    if not os.path.exists(level_dir):
        return None
    
    results = {'level': level, 'threshold': threshold}
    
    # Load nodes
    nodes_file = f"{level_dir}/nodes_level_{level}.csv.gz"
    if os.path.exists(nodes_file):
        df_nodes = pd.read_csv(nodes_file, compression='gzip')
        results['total_nodes'] = len(df_nodes)
        node_ids = set(df_nodes['channel_id'].tolist())
    else:
        results['total_nodes'] = 0
        node_ids = set()
    
    # Load preprocessing stats
    tracking_file = f"{level_dir}/preprocessing/channels_tracking.json"
    if os.path.exists(tracking_file):
        with open(tracking_file, 'r') as f:
            tracking = json.load(f)
        results['channels_found'] = tracking.get('channels_found', 0)
        results['channels_with_english'] = tracking.get('channels_with_english', 0)
        results['channels_too_few_messages'] = tracking.get('channels_too_few_messages', 0)
        results['total_messages'] = tracking.get('total_messages', 0)
        results['english_messages'] = tracking.get('english_messages', 0)
    
    # Load LDA info
    lda_info_file = f"{level_dir}/lda/best_k.json"
    if os.path.exists(lda_info_file):
        with open(lda_info_file, 'r') as f:
            lda_info = json.load(f)
        results['num_topics'] = lda_info.get('best_k', lda_info.get('num_topics', 0))
        results['coherence'] = lda_info.get('best_coherence', 0)
    
    # Load classification results
    gaming_file = f"{level_dir}/classification/gaming_topics.json"
    if os.path.exists(gaming_file):
        with open(gaming_file, 'r') as f:
            gaming_data = json.load(f)
        results['gaming_topics'] = len(gaming_data.get('gaming_topics', []))
        results['total_topics'] = gaming_data.get('total_topics', 0)
    
    # Load channel analysis
    channel_file = f"{level_dir}/channel_analysis/gaming_channels.json"
    if os.path.exists(channel_file):
        with open(channel_file, 'r') as f:
            channel_data = json.load(f)
        results['gaming_channels'] = channel_data.get('gaming_channels', 0)
        results['total_analyzed_channels'] = channel_data.get('total_channels', 0)
        results['gaming_messages'] = channel_data.get('total_gaming_messages', 0)
        gaming_channel_ids = set(channel_data.get('gaming_channel_ids', []))
        results['gaming_channel_ids'] = gaming_channel_ids
    else:
        gaming_channel_ids = set()
        results['gaming_channel_ids'] = set()
    
    # ============================================================
    # METRICHE PER LIVELLO
    # ============================================================
    if gt_gaming_valid is not None:
        predicted_gaming = gaming_channel_ids
        
        gt_gaming_valid_in_level = gt_gaming_valid & node_ids
        gt_non_gaming_valid_in_level = gt_non_gaming_valid & node_ids
        
        tp = len(predicted_gaming & gt_gaming_valid_in_level)
        fp = len(predicted_gaming & gt_non_gaming_valid_in_level)
        fn = len(gt_gaming_valid_in_level - predicted_gaming)
        unlabeled = len(predicted_gaming - all_labeled_channels)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['ground_truth'] = {
            'gt_gaming_valid_in_level': len(gt_gaming_valid_in_level),
            'gt_non_gaming_valid_in_level': len(gt_non_gaming_valid_in_level),
            'predicted_gaming': len(predicted_gaming),
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'unlabeled_predicted': unlabeled,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    return results


# ======================== MAIN ANALYSIS ========================

def analyze_experiment(experiment_name, threshold):
    """Analyze entire experiment."""
    
    base_dir = f"../../results/experiments_tgdataset/{experiment_name}"
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Experiment not found: {base_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f" ANALYSIS: {experiment_name}")
    print(f" Threshold: {threshold*100:.0f}%")
    print(f"{'='*70}")
    
    # Load ground truth
    _, ground_truth_gaming, non_gaming_labeled, all_labeled_channels = load_ground_truth()
    
    # Load valid channels
    gt_gaming_valid, gt_non_gaming_valid = load_valid_channels()
    
    print(f"\nGround truth (TUTTI):")
    print(f"  Gaming: {len(ground_truth_gaming)}")
    print(f"  Non-gaming: {len(non_gaming_labeled)}")
    
    print(f"\nGround truth VALIDI (>= {MIN_CHANNEL_MESSAGES} msg inglesi):")
    print(f"  Gaming validi: {len(gt_gaming_valid)} ({100*len(gt_gaming_valid)/len(ground_truth_gaming):.1f}%)")
    print(f"  Non-gaming validi: {len(gt_non_gaming_valid)} ({100*len(gt_non_gaming_valid)/len(non_gaming_labeled):.1f}%)")
    
    # Analyze each level
    all_results = []
    level = 0
    
    while True:
        result = analyze_level(str(level), base_dir, threshold, gt_gaming_valid, gt_non_gaming_valid, all_labeled_channels)
        if result is None:
            break
        all_results.append(result)
        level += 1
    
    if not all_results:
        print("[ERROR] No levels found!")
        return
    
    # Print per-level summary
    print(f"\n{'='*70}")
    print(f" PER-LEVEL SUMMARY")
    print(f"{'='*70}")
    
    print(f"\n{'Level':<6} {'Nodes':>8} {'Found':>8} {'English':>8} {'Gaming Ch':>10}")
    print("-"*50)
    
    for r in all_results:
        print(f"{r['level']:<6} {r.get('total_nodes', 0):>8} {r.get('channels_found', 0):>8} {r.get('channels_with_english', 0):>8} {r.get('gaming_channels', 0):>10}")
    
    # Ground truth per livello
    print(f"\n{'='*70}")
    print(f" GROUND TRUTH PER LIVELLO")
    print(f"{'='*70}")
    
    print(f"\n{'Level':<6} {'GT Gaming':>10} {'GT Other':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-"*90)
    
    for r in all_results:
        gt = r.get('ground_truth', {})
        if gt:
            print(f"{r['level']:<6} {gt.get('gt_gaming_valid_in_level', 0):>10} {gt.get('gt_non_gaming_valid_in_level', 0):>10} {gt.get('true_positives', 0):>6} {gt.get('false_positives', 0):>6} {gt.get('false_negatives', 0):>6} {gt.get('precision', 0):>7.1%} {gt.get('recall', 0):>7.1%} {gt.get('f1', 0):>7.1%}")
    
    # Global metrics
    print(f"\n{'='*70}")
    print(f" METRICHE GLOBALI")
    print(f"{'='*70}")
    
    all_gaming_ids = set()
    for r in all_results:
        all_gaming_ids.update(r.get('gaming_channel_ids', set()))
    
    global_tp = len(all_gaming_ids & gt_gaming_valid)
    global_fp = len(all_gaming_ids & gt_non_gaming_valid)
    global_fn = len(gt_gaming_valid - all_gaming_ids)
    global_unlabeled = len(all_gaming_ids - all_labeled_channels)
    
    global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
    global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
    global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
    
    print(f"\n  GT gaming validi:          {len(gt_gaming_valid)}")
    print(f"  Predicted gaming (unique): {len(all_gaming_ids)}")
    print(f"  True Positives:            {global_tp}")
    print(f"  False Positives:           {global_fp}")
    print(f"  False Negatives:           {global_fn}")
    print(f"  Unlabeled (esclusi):       {global_unlabeled}")
    print(f"  Precision:                 {global_precision:.1%}")
    print(f"  Recall:                    {global_recall:.1%}")
    print(f"  F1 Score:                  {global_f1:.1%}")
    
    # Save results
    output = {
        'experiment_name': experiment_name,
        'threshold': threshold,
        'min_channel_messages': MIN_CHANNEL_MESSAGES,
        'levels': [{k: (list(v) if isinstance(v, set) else v) for k, v in r.items()} for r in all_results],
        'global': {
            'gt_gaming_valid': len(gt_gaming_valid),
            'gt_non_gaming_valid': len(gt_non_gaming_valid),
            'unique_gaming_channels': len(all_gaming_ids),
            'true_positives': global_tp,
            'false_positives': global_fp,
            'false_negatives': global_fn,
            'precision': global_precision,
            'recall': global_recall,
            'f1': global_f1,
        }
    }
    
    output_file = f"{base_dir}/experiment_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, required=True)
    parser.add_argument('--threshold', type=float, required=True)
    parser.add_argument('--recompute-valid', action='store_true')
    args = parser.parse_args()
    
    if args.recompute_valid and os.path.exists(VALID_CHANNELS_CACHE):
        os.remove(VALID_CHANNELS_CACHE)
        print("Removed valid channels cache.")
    
    analyze_experiment(args.experiment_name, args.threshold)


if __name__ == "__main__":
    main()