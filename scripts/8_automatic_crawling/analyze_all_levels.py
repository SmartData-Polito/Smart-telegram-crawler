#!/usr/bin/env python3
"""
Complete analysis of all levels in the pipeline with detailed metrics.

Usage: 
    python analyze_all_levels.py --threshold 0.4
    python analyze_all_levels.py --threshold 0.4 --experiment-name peak_jul_aug
    python analyze_all_levels.py --thresholds 0.2,0.4,0.6,0.8  # Multiple thresholds
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from glob import glob

def analyze_level(level: str, threshold: float, base_dir: str) -> dict:
    """Analyze a single level and return comprehensive statistics."""
    
    level_dir = f"{base_dir}/level_{level}"
    lda_dir = f"{level_dir}/lda"
    preprocess_dir = f"{level_dir}/preprocessing"
    classification_dir = f"{level_dir}/classification"
    
    print(f"\n{'='*70}")
    print(f"ANALYZING LEVEL: {level} (threshold={threshold*100:.0f}%)")
    print(f"{'='*70}")
    
    # Check if level has been processed
    if not os.path.exists(f"{preprocess_dir}/messages_english_clean.tsv.gz"):
        print(f"  [SKIP] No preprocessed messages found")
        return None
    
    if not os.path.exists(f"{lda_dir}/doc_topic_matrix_level_{level}.npy"):
        print(f"  [SKIP] No doc_topic_matrix found")
        return None
    
    # Load doc_topic_matrix
    print("  Loading data...")
    doc_topic_matrix = np.load(f"{lda_dir}/doc_topic_matrix_level_{level}.npy")
    n_docs, n_topics = doc_topic_matrix.shape
    
    # Load messages
    df_messages = pd.read_csv(f"{preprocess_dir}/messages_english_clean.tsv.gz",
                               sep='\t', compression='gzip')
    
    # Load politics topics
    politics_topics = set()
    politics_path = f"{classification_dir}/politics_topics.json"
    if os.path.exists(politics_path):
        with open(politics_path, 'r') as f:
            politics_data = json.load(f)
            politics_topics = set(politics_data.get("politics_topics", []))
    
    # Load topic keywords
    topics_keywords = {}
    topics_path = f"{level_dir}/topics/topics_for_classification.json"
    if os.path.exists(topics_path):
        with open(topics_path, 'r') as f:
            topics_data = json.load(f)
        
        if isinstance(topics_data, dict) and "topics" in topics_data:
            for t in topics_data["topics"]:
                if isinstance(t, dict):
                    topic_id = t.get('topic_id', len(topics_keywords))
                    topics_keywords[topic_id] = t.get('keywords', [])[:10]
        elif isinstance(topics_data, list):
            for t in topics_data:
                if isinstance(t, dict):
                    topic_id = t.get('topic_id', len(topics_keywords))
                    topics_keywords[topic_id] = t.get('keywords', [])[:10]
    
    # Compute dominant topic and political status for each message
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    df_messages['dominant_topic'] = dominant_topics
    df_messages['is_political_message'] = df_messages['dominant_topic'].isin(politics_topics)
    
    # ==================== GLOBAL METRICS ====================
    total_messages = len(df_messages)
    political_messages = int(df_messages['is_political_message'].sum())
    non_political_messages = total_messages - political_messages
    
    # Channel stats
    channel_stats = df_messages.groupby('channel_id').agg(
        total_messages=('is_political_message', 'count'),
        political_messages=('is_political_message', 'sum')
    ).reset_index()
    
    channel_stats['non_political_messages'] = channel_stats['total_messages'] - channel_stats['political_messages']
    channel_stats['political_ratio'] = channel_stats['political_messages'] / channel_stats['total_messages']
    channel_stats['is_political_channel'] = channel_stats['political_ratio'] >= threshold
    
    total_groups = len(channel_stats)
    political_groups = int(channel_stats['is_political_channel'].sum())
    non_political_groups = total_groups - political_groups
    
    # ==================== POLITICAL GROUPS METRICS ====================
    df_pol_groups = channel_stats[channel_stats['is_political_channel']]
    
    if len(df_pol_groups) > 0:
        avg_msgs_per_pol_group = df_pol_groups['total_messages'].mean()
        avg_pol_msgs_per_pol_group = df_pol_groups['political_messages'].mean()
        avg_non_pol_msgs_per_pol_group = df_pol_groups['non_political_messages'].mean()
    else:
        avg_msgs_per_pol_group = 0
        avg_pol_msgs_per_pol_group = 0
        avg_non_pol_msgs_per_pol_group = 0
    
    # ==================== NON-POLITICAL GROUPS METRICS ====================
    df_non_pol_groups = channel_stats[~channel_stats['is_political_channel']]
    
    if len(df_non_pol_groups) > 0:
        avg_msgs_per_non_pol_group = df_non_pol_groups['total_messages'].mean()
        avg_pol_msgs_per_non_pol_group = df_non_pol_groups['political_messages'].mean()
        avg_non_pol_msgs_per_non_pol_group = df_non_pol_groups['non_political_messages'].mean()
    else:
        avg_msgs_per_non_pol_group = 0
        avg_pol_msgs_per_non_pol_group = 0
        avg_non_pol_msgs_per_non_pol_group = 0
    
    # ==================== TOPIC-LEVEL METRICS ====================
    topic_stats = []
    total_pol_msgs_in_topics = 0
    total_non_pol_msgs_in_topics = 0
    
    for topic_id in range(n_topics):
        topic_mask = df_messages['dominant_topic'] == topic_id
        topic_messages = int(topic_mask.sum())
        
        is_political_topic = topic_id in politics_topics
        keywords = topics_keywords.get(topic_id, [])
        
        if is_political_topic:
            total_pol_msgs_in_topics += topic_messages
        else:
            total_non_pol_msgs_in_topics += topic_messages
        
        topic_stats.append({
            "topic_id": topic_id,
            "is_political_topic": is_political_topic,
            "keywords": keywords,
            "total_messages": topic_messages
        })
    
    n_political_topics = len(politics_topics)
    n_non_political_topics = n_topics - n_political_topics
    
    avg_msgs_per_pol_topic = total_pol_msgs_in_topics / n_political_topics if n_political_topics > 0 else 0
    avg_msgs_per_non_pol_topic = total_non_pol_msgs_in_topics / n_non_political_topics if n_non_political_topics > 0 else 0
    
    # Sort topics by message count
    topic_stats = sorted(topic_stats, key=lambda x: x['total_messages'], reverse=True)
    
    # ==================== PRINT RESULTS ====================
    print(f"\n  GLOBAL METRICS:")
    print(f"    Total messages:              {total_messages:>10,}")
    print(f"    Political messages:          {political_messages:>10,} ({100*political_messages/total_messages:.1f}%)")
    print(f"    Non-political messages:      {non_political_messages:>10,} ({100*non_political_messages/total_messages:.1f}%)")
    print(f"    Total groups:                {total_groups:>10,}")
    print(f"    Political groups:            {political_groups:>10,} ({100*political_groups/total_groups:.1f}%)")
    print(f"    Non-political groups:        {non_political_groups:>10,} ({100*non_political_groups/total_groups:.1f}%)")
    
    print(f"\n  POLITICAL GROUPS (>={threshold*100:.0f}% political messages):")
    print(f"    Avg messages per group:      {avg_msgs_per_pol_group:>10.1f}")
    print(f"    Avg political msgs/group:    {avg_pol_msgs_per_pol_group:>10.1f}")
    print(f"    Avg non-political msgs/group:{avg_non_pol_msgs_per_pol_group:>10.1f}")
    
    print(f"\n  NON-POLITICAL GROUPS (<{threshold*100:.0f}% political messages):")
    print(f"    Avg messages per group:      {avg_msgs_per_non_pol_group:>10.1f}")
    print(f"    Avg political msgs/group:    {avg_pol_msgs_per_non_pol_group:>10.1f}")
    print(f"    Avg non-political msgs/group:{avg_non_pol_msgs_per_non_pol_group:>10.1f}")
    
    print(f"\n  TOPIC-LEVEL METRICS:")
    print(f"    Total topics:                {n_topics:>10,}")
    print(f"    Political topics:            {n_political_topics:>10,}")
    print(f"    Non-political topics:        {n_non_political_topics:>10,}")
    print(f"    Avg msgs per political topic:    {avg_msgs_per_pol_topic:>10.1f}")
    print(f"    Avg msgs per non-pol topic:      {avg_msgs_per_non_pol_topic:>10.1f}")
    
    print(f"\n  TOP 5 TOPICS:")
    for t in topic_stats[:5]:
        pol_marker = "[P]" if t['is_political_topic'] else "[ ]"
        kw_preview = ", ".join(t['keywords'][:5]) if t['keywords'] else "N/A"
        print(f"    Topic {t['topic_id']:3} {pol_marker}: {t['total_messages']:>6,} msgs - {kw_preview}")
    
    # ==================== BUILD RESULT ====================
    result = {
        "level": level,
        "threshold": threshold,
        
        "global_metrics": {
            "total_messages": total_messages,
            "political_messages": political_messages,
            "non_political_messages": non_political_messages,
            "total_groups": total_groups,
            "political_groups": political_groups,
            "non_political_groups": non_political_groups
        },
        
        "political_groups_metrics": {
            "count": political_groups,
            "avg_messages_per_group": round(avg_msgs_per_pol_group, 2),
            "avg_political_messages_per_group": round(avg_pol_msgs_per_pol_group, 2),
            "avg_non_political_messages_per_group": round(avg_non_pol_msgs_per_pol_group, 2)
        },
        
        "non_political_groups_metrics": {
            "count": non_political_groups,
            "avg_messages_per_group": round(avg_msgs_per_non_pol_group, 2),
            "avg_political_messages_per_group": round(avg_pol_msgs_per_non_pol_group, 2),
            "avg_non_political_messages_per_group": round(avg_non_pol_msgs_per_non_pol_group, 2)
        },
        
        "topic_level_metrics": {
            "total_topics": n_topics,
            "political_topics": n_political_topics,
            "non_political_topics": n_non_political_topics,
            "avg_messages_per_political_topic": round(avg_msgs_per_pol_topic, 2),
            "avg_messages_per_non_political_topic": round(avg_msgs_per_non_pol_topic, 2)
        },
        
        "topics": topic_stats
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.4,
                        help="Single threshold (default: 0.4)")
    parser.add_argument("--thresholds", type=str, default=None,
                        help="Comma-separated thresholds (e.g., '0.2,0.4,0.6,0.8')")
    parser.add_argument("--level", type=str, default=None,
                        help="Analyze specific level only")
    parser.add_argument("--experiment-name", type=str, default=None,
                        help="Experiment name (folder in results/experiments/)")
    parser.add_argument("--base-dir", type=str, default=None,
                        help="Override base directory")
    args = parser.parse_args()
    
    # Determine base directory
    if args.base_dir:
        base_dir = args.base_dir
    elif args.experiment_name:
        base_dir = f"../../results/experiments/{args.experiment_name}"
    else:
        base_dir = "../../results/levels_automatic"
    
    # Determine thresholds
    if args.thresholds:
        thresholds = [float(t.strip()) for t in args.thresholds.split(',')]
    else:
        thresholds = [args.threshold]
    
    # Determine output directory
    output_dir = f"{base_dir}/analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=" * 70)
    print(f"COMPLETE PIPELINE ANALYSIS")
    print(f"Base dir: {base_dir}")
    print(f"Thresholds: {[f'{t*100:.0f}%' for t in thresholds]}")
    print(f"=" * 70)
    
    # Find all levels
    if args.level is not None:
        levels = [args.level]
    else:
        level_dirs = glob(f"{base_dir}/level_*")
        levels = []
        for d in level_dirs:
            level_name = os.path.basename(d).replace("level_", "")
            levels.append(level_name)
        
        numeric_levels = sorted([l for l in levels if l.isdigit()], key=int)
        other_levels = sorted([l for l in levels if not l.isdigit()])
        levels = numeric_levels + other_levels
    
    print(f"Levels to analyze: {levels}")
    
    # Analyze for each threshold
    all_results = {}
    
    for threshold in thresholds:
        print(f"\n{'#'*70}")
        print(f"# THRESHOLD: {threshold*100:.0f}%")
        print(f"{'#'*70}")
        
        threshold_results = {}
        summary_rows = []
        
        for level in levels:
            result = analyze_level(level, threshold, base_dir)
            
            if result is not None:
                threshold_results[level] = result
                
                summary_rows.append({
                    "level": level,
                    "threshold": threshold,
                    **result["global_metrics"],
                    "avg_msgs_per_pol_group": result["political_groups_metrics"]["avg_messages_per_group"],
                    "avg_pol_msgs_per_pol_group": result["political_groups_metrics"]["avg_political_messages_per_group"],
                    "avg_msgs_per_non_pol_group": result["non_political_groups_metrics"]["avg_messages_per_group"],
                    "avg_msgs_per_pol_topic": result["topic_level_metrics"]["avg_messages_per_political_topic"],
                    "avg_msgs_per_non_pol_topic": result["topic_level_metrics"]["avg_messages_per_non_political_topic"]
                })
        
        all_results[f"threshold_{int(threshold*100)}"] = threshold_results
        
        # Save threshold-specific results
        if summary_rows:
            df_summary = pd.DataFrame(summary_rows)
            threshold_csv = f"{output_dir}/summary_threshold_{int(threshold*100)}.csv"
            df_summary.to_csv(threshold_csv, index=False)
            print(f"\nSaved: {threshold_csv}")
    
    # Save complete analysis
    complete_output = f"{output_dir}/complete_analysis.json"
    with open(complete_output, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved complete analysis: {complete_output}")
    
    # Print final comparison table
    print(f"\n{'='*70}")
    print("FINAL COMPARISON (all thresholds)")
    print(f"{'='*70}")
    
    # Aggregate by threshold
    for threshold in thresholds:
        threshold_key = f"threshold_{int(threshold*100)}"
        if threshold_key in all_results:
            results = all_results[threshold_key]
            
            total_msgs = sum(r["global_metrics"]["total_messages"] for r in results.values())
            total_pol_msgs = sum(r["global_metrics"]["political_messages"] for r in results.values())
            total_groups = sum(r["global_metrics"]["total_groups"] for r in results.values())
            total_pol_groups = sum(r["global_metrics"]["political_groups"] for r in results.values())
            
            print(f"\nThreshold {threshold*100:.0f}%:")
            print(f"  Total messages: {total_msgs:,}")
            print(f"  Political messages: {total_pol_msgs:,} ({100*total_pol_msgs/total_msgs:.1f}%)")
            print(f"  Total groups: {total_groups:,}")
            print(f"  Political groups: {total_pol_groups:,} ({100*total_pol_groups/total_groups:.1f}%)")

if __name__ == "__main__":
    main()