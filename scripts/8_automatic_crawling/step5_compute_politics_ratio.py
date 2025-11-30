#!/usr/bin/env python3
"""
STEP 5: Compute political message percentage per channel.
Usage: python step5_compute_politics_ratio.py --level 0 --threshold 0.4

This script:
1. Loads document-topic assignments
2. Uses classified politics topics
3. Computes % of political messages per channel
4. Identifies channels meeting threshold for next level
"""

import os
import time
import argparse
import json
import numpy as np
import pandas as pd
import joblib

# ======================== CONFIGURATION ========================
DEFAULT_POLITICS_THRESHOLD = 0.40  # 40% minimum political messages

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Compute political ratio per channel")
    parser.add_argument("--level", type=str, required=True, help="Current hierarchy level")
    parser.add_argument("--threshold", type=float, default=DEFAULT_POLITICS_THRESHOLD,
                        help=f"Minimum political ratio (default: {DEFAULT_POLITICS_THRESHOLD})")
    args = parser.parse_args()
    
    level = args.level
    threshold = args.threshold
    log_time(f"Computing political ratios for level {level} (threshold={threshold})")
    
    # Paths
    base_dir = f"../results/levels_automatic/level_{level}"
    lda_dir = f"{base_dir}/lda"
    preprocess_dir = f"{base_dir}/preprocessing"
    
    # Input files
    politics_topics_path = f"{lda_dir}/politics_topics.json"
    assignments_path = f"{lda_dir}/doc_topic_assignments.csv.gz"
    messages_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    
    # Output files
    channel_stats_path = f"{lda_dir}/channel_politics_stats.csv"
    political_channels_path = f"{lda_dir}/political_channels.json"
    
    # Load politics topics classification
    if not os.path.exists(politics_topics_path):
        log_time(f"ERROR: Politics topics not found: {politics_topics_path}")
        return
    
    with open(politics_topics_path, "r") as f:
        politics_data = json.load(f)
    
    politics_topics = set(politics_data["politics_topics"])
    log_time(f"Loaded {len(politics_topics)} politics topics: {sorted(politics_topics)}")
    
    # Load document-topic assignments
    log_time("Loading document-topic assignments...")
    df_assignments = pd.read_csv(assignments_path, compression='gzip')
    log_time(f"Loaded {len(df_assignments)} document assignments")
    
    # Mark each document as political or not (based on dominant topic)
    df_assignments['is_political'] = df_assignments['dominant_topic'].isin(politics_topics)
    
    # Compute statistics per channel
    log_time("Computing per-channel statistics...")
    
    channel_stats = df_assignments.groupby('channel_id').agg(
        total_messages=('is_political', 'count'),
        political_messages=('is_political', 'sum'),
        avg_topic_confidence=('max_topic_score', 'mean')
    ).reset_index()
    
    channel_stats['political_ratio'] = (
        channel_stats['political_messages'] / channel_stats['total_messages']
    )
    
    # Sort by political ratio
    channel_stats = channel_stats.sort_values('political_ratio', ascending=False)
    
    # Save full stats
    channel_stats.to_csv(channel_stats_path, index=False)
    log_time(f"Saved channel stats to {channel_stats_path}")
    
    # Identify political channels (meeting threshold)
    political_channels = channel_stats[
        channel_stats['political_ratio'] >= threshold
    ]['channel_id'].tolist()
    
    non_political_channels = channel_stats[
        channel_stats['political_ratio'] < threshold
    ]['channel_id'].tolist()
    
    log_time(f"Channels meeting threshold ({threshold}): {len(political_channels)}")
    log_time(f"Channels below threshold: {len(non_political_channels)}")
    
    # Save political channels list
    output_data = {
        "level": level,
        "threshold": threshold,
        "total_channels": len(channel_stats),
        "political_channels_count": len(political_channels),
        "non_political_channels_count": len(non_political_channels),
        "political_channels": political_channels,
        "non_political_channels": non_political_channels
    }
    
    with open(political_channels_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log_time(f"Saved political channels to {political_channels_path}")
    
    # Print summary statistics
    log_time("=" * 50)
    log_time("SUMMARY STATISTICS:")
    log_time(f"  Total channels: {len(channel_stats)}")
    log_time(f"  Political (>={threshold*100:.0f}%): {len(political_channels)}")
    log_time(f"  Non-political (<{threshold*100:.0f}%): {len(non_political_channels)}")
    log_time(f"  Mean political ratio: {channel_stats['political_ratio'].mean():.2%}")
    log_time(f"  Median political ratio: {channel_stats['political_ratio'].median():.2%}")
    
    # Show top political channels
    log_time("\nTop 10 most political channels:")
    top10 = channel_stats.head(10)
    for _, row in top10.iterrows():
        log_time(f"  {row['channel_id']}: {row['political_ratio']:.1%} ({row['political_messages']}/{row['total_messages']})")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    log_time(f"\nCOMPLETED in {total_time:.2f}s")
    
    with open(f"{lda_dir}/step5_completed.txt", "w") as f:
        f.write(f"Politics ratio computation completed in {total_time:.2f}s\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Political channels: {len(political_channels)}\n")
        f.write(f"Non-political channels: {len(non_political_channels)}\n")

if __name__ == "__main__":
    main()