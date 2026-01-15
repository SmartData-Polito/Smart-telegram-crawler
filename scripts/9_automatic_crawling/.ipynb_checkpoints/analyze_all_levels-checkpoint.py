#!/usr/bin/env python3
"""
Analyze results from TGDataset gaming detection pipeline.
Computes metrics, compares with ground truth, generates reports.

METRICHE PER LIVELLO (calcolate sui canali DEL LIVELLO):
- GT Gaming in level = canali del livello con etichetta "Videogame modding"
- GT Other in level = canali del livello con etichetta DIVERSA
- TP = canali del livello con etichetta "Videogame modding" E classificati gaming
- FP = canali del livello con etichetta DIVERSA E classificati gaming
- FN = canali del livello con etichetta "Videogame modding" E NON classificati
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

METRICHE GLOBALI (calcolate su TUTTI i 1957 canali gaming):
- TP = canali con etichetta "Videogame modding" classificati gaming (in qualsiasi livello)
- FP = canali con etichetta DIVERSA classificati gaming
- FN = TUTTI i 1957 "Videogame modding" NON classificati
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)

Canali senza etichetta sono ESCLUSI da tutte le metriche.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from collections import defaultdict

# ======================== CONFIGURATION ========================
TGDATASET_DIR = "../../material"
LABELED_DATA_DIR = f"{TGDATASET_DIR}/TGDataset/labeled_data"

# ======================== GROUND TRUTH ========================
def load_ground_truth():
    """Load ground truth labels from ch_to_topic_mapping.csv
    
    Returns:
        df: DataFrame with all labels
        gaming_channels: set of channel IDs labeled as "Videogame modding"
        non_gaming_labeled: set of channel IDs labeled with OTHER topics
        all_labeled_channels: set of ALL channel IDs that have ANY label
    """
    topic_file = f"{LABELED_DATA_DIR}/ch_to_topic_mapping.csv"
    
    if not os.path.exists(topic_file):
        print(f"[WARN] Ground truth not found: {topic_file}")
        return None, set(), set(), set()
    
    df = pd.read_csv(topic_file)
    
    # Gaming = solo "Videogame modding"
    gaming_channels = set(df[df['topic'] == 'Videogame modding']['ch_ID'].tolist())
    
    # TUTTI i canali che hanno UNA QUALSIASI etichetta
    all_labeled_channels = set(df['ch_ID'].tolist())
    
    # Canali etichettati con topic DIVERSO da gaming
    non_gaming_labeled = all_labeled_channels - gaming_channels
    
    return df, gaming_channels, non_gaming_labeled, all_labeled_channels

# ======================== LEVEL ANALYSIS ========================
def analyze_level(level, base_dir, threshold, ground_truth_gaming=None, non_gaming_labeled=None, all_labeled_channels=None):
    """Analyze a single level."""
    
    level_dir = f"{base_dir}/level_{level}"
    
    if not os.path.exists(level_dir):
        return None
    
    results = {
        'level': level,
        'threshold': threshold,
    }
    
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
        results['total_messages'] = tracking.get('total_messages', 0)
        results['english_messages'] = tracking.get('english_messages', 0)
        results['forwarded_messages'] = tracking.get('total_forwarded_messages', 0)
    
    # Load LDA info
    lda_info_file = f"{level_dir}/lda/best_k.json"
    if os.path.exists(lda_info_file):
        with open(lda_info_file, 'r') as f:
            lda_info = json.load(f)
        results['num_topics'] = lda_info.get('best_k', lda_info.get('num_topics', 0))
        results['coherence'] = lda_info.get('best_coherence', 0)
    
    # Load classification results
    gaming_file = f"{level_dir}/classification/gaming_topics.json"
    if not os.path.exists(gaming_file):
        gaming_file = f"{level_dir}/classification/politics_topics.json"
    
    if os.path.exists(gaming_file):
        with open(gaming_file, 'r') as f:
            gaming_data = json.load(f)
        results['gaming_topics'] = len(gaming_data.get('gaming_topics', gaming_data.get('politics_topics', [])))
        results['total_topics'] = gaming_data.get('total_topics', 0)
    
    # Load channel analysis
    channel_file = f"{level_dir}/channel_analysis/gaming_channels.json"
    if not os.path.exists(channel_file):
        channel_file = f"{level_dir}/channel_analysis/political_channels.json"
    
    if os.path.exists(channel_file):
        with open(channel_file, 'r') as f:
            channel_data = json.load(f)
        results['gaming_channels'] = channel_data.get('gaming_channels', channel_data.get('political_channels', 0))
        results['total_analyzed_channels'] = channel_data.get('total_channels', 0)
        results['gaming_messages'] = channel_data.get('total_gaming_messages', channel_data.get('total_political_messages', 0))
        results['mean_gaming_ratio'] = channel_data.get('mean_gaming_ratio', channel_data.get('mean_political_ratio', 0))
        
        gaming_channel_ids = set(channel_data.get('gaming_channel_ids', channel_data.get('political_channel_ids', [])))
        results['gaming_channel_ids'] = gaming_channel_ids
    else:
        gaming_channel_ids = set()
        results['gaming_channel_ids'] = set()
    
    # Load expansion info (for next level)
    expansion_file = f"{base_dir}/level_{int(level)+1}/expansion_info.json"
    if os.path.exists(expansion_file):
        with open(expansion_file, 'r') as f:
            expansion = json.load(f)
        results['next_level_candidates'] = expansion.get('unique_forwarded_from', 0)
        results['next_level_new'] = expansion.get('new_channels', 0)
    
    # ============================================================
    # METRICHE PER LIVELLO: calcolate sui canali DEL LIVELLO
    # ============================================================
    if ground_truth_gaming is not None and non_gaming_labeled is not None and all_labeled_channels is not None:
        
        # Predizioni del crawler per questo livello
        predicted_gaming = gaming_channel_ids
        
        # GT gaming presenti IN QUESTO LIVELLO
        gt_gaming_in_level = ground_truth_gaming & node_ids
        
        # GT non-gaming (etichettati altro) presenti IN QUESTO LIVELLO
        gt_non_gaming_in_level = non_gaming_labeled & node_ids
        
        # TP: canali del livello con etichetta gaming E classificati gaming
        true_positives = len(predicted_gaming & gt_gaming_in_level)
        
        # FP: canali del livello con etichetta ALTRO E classificati gaming
        false_positives = len(predicted_gaming & gt_non_gaming_in_level)
        
        # FN: canali del livello con etichetta gaming E NON classificati
        false_negatives = len(gt_gaming_in_level - predicted_gaming)
        
        # Canali predetti gaming ma senza etichetta (per report, esclusi dalle metriche)
        unlabeled_predicted = len(predicted_gaming - all_labeled_channels)
        
        # Precision: TP / (TP + FP) - sul livello
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        
        # Recall: TP / (TP + FN) - sul livello
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # F1
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results['ground_truth'] = {
            'gt_gaming_in_level': len(gt_gaming_in_level),
            'gt_non_gaming_in_level': len(gt_non_gaming_in_level),
            'predicted_gaming': len(predicted_gaming),
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'unlabeled_predicted': unlabeled_predicted,
            'precision': precision,
            'recall': recall,
            'f1': f1,
        }
    
    return results

# ======================== GLOBAL ANALYSIS ========================
def analyze_experiment(experiment_name, threshold):
    """Analyze entire experiment across all levels."""
    
    base_dir = f"../../results/experiments_tgdataset/{experiment_name}"
    
    if not os.path.exists(base_dir):
        print(f"[ERROR] Experiment not found: {base_dir}")
        return
    
    print(f"\n{'='*70}")
    print(f" ANALYSIS: {experiment_name}")
    print(f" Threshold: {threshold*100:.0f}%")
    print(f"{'='*70}")
    
    # Load ground truth
    df_labels, ground_truth_gaming, non_gaming_labeled, all_labeled_channels = load_ground_truth()
    if ground_truth_gaming:
        print(f"\nGround truth:")
        print(f"  Canali 'Videogame modding': {len(ground_truth_gaming)}")
        print(f"  Canali con altre etichette: {len(non_gaming_labeled)}")
        print(f"  Totale etichettati: {len(all_labeled_channels)}")
    
    # Load seed info
    seed_file = f"{base_dir}/level_0/seed_info.json"
    if os.path.exists(seed_file):
        with open(seed_file, 'r') as f:
            seed_info = json.load(f)
        print(f"\nSeed info:")
        print(f"  Total seeds: {seed_info.get('total_seeds', 'N/A')}")
        print(f"  Gaming seeds: {seed_info.get('gaming_seeds', 'N/A')}")
        print(f"  Non-gaming seeds: {seed_info.get('non_gaming_seeds', 'N/A')}")
    
    # Analyze each level
    all_results = []
    level = 0
    
    while True:
        result = analyze_level(str(level), base_dir, threshold, ground_truth_gaming, non_gaming_labeled, all_labeled_channels)
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
    
    print(f"\n{'Level':<6} {'Nodes':>8} {'Found':>8} {'English':>8} {'Gaming Ch':>10} {'Gaming %':>10}")
    print("-"*70)
    
    for r in all_results:
        gaming_pct = r.get('gaming_channels', 0) / r.get('total_analyzed_channels', 1) * 100 if r.get('total_analyzed_channels') else 0
        print(f"{r['level']:<6} {r.get('total_nodes', 0):>8} {r.get('channels_found', 0):>8} {r.get('channels_with_english', 0):>8} {r.get('gaming_channels', 0):>10} {gaming_pct:>9.1f}%")
    
    # Messages summary
    print(f"\n{'Level':<6} {'Total Msgs':>12} {'English':>12} {'Gaming':>12} {'Gaming %':>10}")
    print("-"*70)
    
    for r in all_results:
        gaming_msg_pct = r.get('gaming_messages', 0) / r.get('english_messages', 1) * 100 if r.get('english_messages') else 0
        print(f"{r['level']:<6} {r.get('total_messages', 0):>12,} {r.get('english_messages', 0):>12,} {r.get('gaming_messages', 0):>12,} {gaming_msg_pct:>9.1f}%")
    
    # Topics summary
    print(f"\n{'Level':<6} {'Topics K':>10} {'Gaming Topics':>14} {'Gaming %':>10} {'Coherence':>10}")
    print("-"*70)
    
    for r in all_results:
        gaming_topic_pct = r.get('gaming_topics', 0) / r.get('num_topics', 1) * 100 if r.get('num_topics') else 0
        print(f"{r['level']:<6} {r.get('num_topics', 0):>10} {r.get('gaming_topics', 0):>14} {gaming_topic_pct:>9.1f}% {r.get('coherence', 0):>10.4f}")
    
    # Ground truth comparison PER LIVELLO
    if ground_truth_gaming:
        print(f"\n{'='*70}")
        print(f" GROUND TRUTH PER LIVELLO (metriche sui canali del livello)")
        print(f"{'='*70}")
        
        print(f"\n{'Level':<6} {'GT Gaming':>10} {'GT Other':>10} {'Predicted':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Unlbl':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-"*110)
        
        for r in all_results:
            gt = r.get('ground_truth', {})
            if gt:
                print(f"{r['level']:<6} {gt.get('gt_gaming_in_level', 0):>10} {gt.get('gt_non_gaming_in_level', 0):>10} {gt.get('predicted_gaming', 0):>10} {gt.get('true_positives', 0):>6} {gt.get('false_positives', 0):>6} {gt.get('false_negatives', 0):>6} {gt.get('unlabeled_predicted', 0):>6} {gt.get('precision', 0):>7.1%} {gt.get('recall', 0):>7.1%} {gt.get('f1', 0):>7.1%}")
    
    # Global metrics
    print(f"\n{'='*70}")
    print(f" GLOBAL METRICS")
    print(f"{'='*70}")
    
    total_nodes = sum(r.get('total_nodes', 0) for r in all_results)
    total_channels_found = sum(r.get('channels_found', 0) for r in all_results)
    total_english = sum(r.get('channels_with_english', 0) for r in all_results)
    total_gaming = sum(r.get('gaming_channels', 0) for r in all_results)
    total_messages = sum(r.get('total_messages', 0) for r in all_results)
    total_english_msgs = sum(r.get('english_messages', 0) for r in all_results)
    total_gaming_msgs = sum(r.get('gaming_messages', 0) for r in all_results)
    
    print(f"\n  Levels processed:          {len(all_results)}")
    print(f"  Total nodes:               {total_nodes:,}")
    print(f"  Channels found:            {total_channels_found:,}")
    print(f"  Channels with English:     {total_english:,}")
    print(f"  Gaming channels:           {total_gaming:,} ({total_gaming/total_english*100:.1f}%)" if total_english else "  Gaming channels:           0")
    print(f"  Total messages:            {total_messages:,}")
    print(f"  English messages:          {total_english_msgs:,}")
    print(f"  Gaming messages:           {total_gaming_msgs:,} ({total_gaming_msgs/total_english_msgs*100:.1f}%)" if total_english_msgs else "  Gaming messages:           0")
    
    # Unique gaming channels across all levels
    all_gaming_ids = set()
    for r in all_results:
        all_gaming_ids.update(r.get('gaming_channel_ids', set()))
    
    print(f"\n  Unique gaming channels:    {len(all_gaming_ids)}")
    
    if ground_truth_gaming:
        # ============================================================
        # METRICHE GLOBALI: rispetto a TUTTI i 1957 canali gaming
        # ============================================================
        
        # TP: etichetta "Videogame modding" E classificato gaming (in qualsiasi livello)
        global_tp = len(all_gaming_ids & ground_truth_gaming)
        
        # FP: etichetta ALTRO E classificato gaming
        global_fp = len(all_gaming_ids & non_gaming_labeled)
        
        # FN: TUTTI i 1957 gaming NON classificati
        global_fn = len(ground_truth_gaming - all_gaming_ids)
        
        # Canali predetti gaming ma senza etichetta
        global_unlabeled = len(all_gaming_ids - all_labeled_channels)
        
        # Precision: TP / (TP + FP)
        global_precision = global_tp / (global_tp + global_fp) if (global_tp + global_fp) > 0 else 0
        
        # Recall: TP / (TP + FN) - rispetto a TUTTI i 1957
        global_recall = global_tp / (global_tp + global_fn) if (global_tp + global_fn) > 0 else 0
        
        # F1
        global_f1 = 2 * global_precision * global_recall / (global_precision + global_recall) if (global_precision + global_recall) > 0 else 0
        
        print(f"\n  --- GLOBAL Ground Truth (rispetto a TUTTI i {len(ground_truth_gaming)} gaming) ---")
        print(f"  Total GT gaming:           {len(ground_truth_gaming)}")
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
        'levels': all_results,
        'global': {
            'total_nodes': total_nodes,
            'channels_found': total_channels_found,
            'channels_with_english': total_english,
            'gaming_channels': total_gaming,
            'unique_gaming_channels': len(all_gaming_ids),
            'total_messages': total_messages,
            'english_messages': total_english_msgs,
            'gaming_messages': total_gaming_msgs,
        }
    }
    
    if ground_truth_gaming:
        output['global']['ground_truth'] = {
            'total_gt_gaming': len(ground_truth_gaming),
            'true_positives': global_tp,
            'false_positives': global_fp,
            'false_negatives': global_fn,
            'unlabeled_predicted': global_unlabeled,
            'precision': global_precision,
            'recall': global_recall,
            'f1': global_f1,
        }
    
    # Convert sets to lists for JSON serialization
    for r in output['levels']:
        if 'gaming_channel_ids' in r:
            r['gaming_channel_ids'] = list(r['gaming_channel_ids'])
    
    output_file = f"{base_dir}/experiment_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {output_file}")
    
    return output

# ======================== COMPARE EXPERIMENTS ========================
def compare_experiments(experiment_names):
    """Compare multiple experiments."""
    
    print(f"\n{'='*70}")
    print(f" EXPERIMENT COMPARISON (Global Metrics)")
    print(f"{'='*70}")
    
    results = []
    
    for exp_name in experiment_names:
        analysis_file = f"../../results/experiments_tgdataset/{exp_name}/experiment_analysis.json"
        if os.path.exists(analysis_file):
            with open(analysis_file, 'r') as f:
                results.append(json.load(f))
        else:
            print(f"[WARN] Analysis not found for {exp_name}, run analyze first")
    
    if not results:
        return
    
    # Summary table
    print(f"\n{'Experiment':<30} {'Levels':>7} {'Channels':>10} {'Gaming':>10} {'TP':>6} {'FP':>6} {'FN':>6} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    print("-"*115)
    
    for r in results:
        g = r.get('global', {})
        gt = g.get('ground_truth', {})
        
        print(f"{r['experiment_name']:<30} {len(r.get('levels', [])):>7} {g.get('channels_with_english', 0):>10,} {g.get('gaming_channels', 0):>10,} {gt.get('true_positives', 0):>6} {gt.get('false_positives', 0):>6} {gt.get('false_negatives', 0):>6} {gt.get('precision', 0):>7.1%} {gt.get('recall', 0):>7.1%} {gt.get('f1', 0):>7.1%}")

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, default=None,
                        help='Analyze specific experiment')
    parser.add_argument('--threshold', type=float, default=0.4,
                        help='Threshold used in experiment')
    parser.add_argument('--level', type=str, default=None,
                        help='Analyze specific level only')
    parser.add_argument('--compare', nargs='+', default=None,
                        help='Compare multiple experiments')
    parser.add_argument('--all', action='store_true',
                        help='Analyze all experiments')
    args = parser.parse_args()
    
    if args.compare:
        compare_experiments(args.compare)
    elif args.all:
        # Find all experiments
        exp_dir = "../../results/experiments_tgdataset"
        if os.path.exists(exp_dir):
            experiments = [d for d in os.listdir(exp_dir) 
                          if os.path.isdir(f"{exp_dir}/{d}") and d.startswith("threshold_")]
            
            for exp in sorted(experiments):
                # Extract threshold from name
                parts = exp.split('_')
                threshold = int(parts[1]) / 100 if len(parts) > 1 else 0.4
                analyze_experiment(exp, threshold)
            
            # Compare all
            if experiments:
                compare_experiments(sorted(experiments))
    elif args.experiment_name:
        analyze_experiment(args.experiment_name, args.threshold)
    else:
        print("Usage:")
        print("  python analyze_all_levels.py --experiment-name threshold_40_pure --threshold 0.4")
        print("  python analyze_all_levels.py --all")
        print("  python analyze_all_levels.py --compare threshold_40_pure threshold_40_mixed")

if __name__ == "__main__":
    main()