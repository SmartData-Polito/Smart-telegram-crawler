#!/usr/bin/env python3
"""
STEP 5: Compute political message percentage per channel.

Usage: python step5_compute_politics_ratio.py --level 0 --threshold 0.4

Output: channel_analysis/

MODIFICHE:
- Rimosso calcolo doc_topic_matrix, ora caricata da lda/doc_topic_matrix_level_{level}.npy
- Rimosso caricamento di lda_model e dictionary (non più necessari)
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd

# ======================== CONFIGURATION ========================
DEFAULT_POLITICS_THRESHOLD = 0.40

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
    base_dir = f"../../results/levels_automatic/level_{level}"
    lda_dir = f"{base_dir}/lda"
    preprocess_dir = f"{base_dir}/preprocessing"
    classification_dir = f"{base_dir}/classification"
    channel_analysis_dir = f"{base_dir}/channel_analysis"
    os.makedirs(channel_analysis_dir, exist_ok=True)

    # Input files
    politics_topics_path = f"{classification_dir}/politics_topics.json"
    doc_topic_matrix_path = f"{lda_dir}/doc_topic_matrix_level_{level}.npy"
    messages_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"

    # Output files
    channel_stats_path = f"{channel_analysis_dir}/channel_politics_stats.csv"
    political_channels_path = f"{channel_analysis_dir}/political_channels.json"
    assignments_path = f"{channel_analysis_dir}/doc_topic_assignments.csv.gz"

    # ==================== Load politics topics ====================
    if not os.path.exists(politics_topics_path):
        log_time(f"ERROR: Politics topics not found: {politics_topics_path}")
        log_time("Run step4_classify_topics.py first")
        sys.exit(1)

    with open(politics_topics_path, "r") as f:
        politics_data = json.load(f)
    politics_topics = set(politics_data["politics_topics"])
    log_time(f"Loaded {len(politics_topics)} politics topics: {sorted(politics_topics)}")

    # ==================== Load pre-computed doc_topic_matrix ====================
    if not os.path.exists(doc_topic_matrix_path):
        log_time(f"ERROR: doc_topic_matrix not found: {doc_topic_matrix_path}")
        log_time("Run step3_extract_topics.py first")
        sys.exit(1)

    log_time("Loading pre-computed doc_topic_matrix...")
    doc_topic_matrix = np.load(doc_topic_matrix_path)
    log_time(f"doc_topic_matrix shape: {doc_topic_matrix.shape}")

    # ==================== Load messages (only channel_id) ====================
    if not os.path.exists(messages_path):
        log_time(f"ERROR: Messages file not found: {messages_path}")
        sys.exit(1)

    log_time("Loading messages...")
    df_messages = pd.read_csv(messages_path, sep='\t', compression='gzip',
                               usecols=['channel_id', 'text_lda'])
    df_messages = df_messages[df_messages['text_lda'].astype(str).str.strip() != '']
    log_time(f"Loaded {len(df_messages)} messages")

    # Verifica che le dimensioni corrispondano
    if len(df_messages) != doc_topic_matrix.shape[0]:
        log_time(f"ERROR: Mismatch! Messages: {len(df_messages)}, doc_topic_matrix rows: {doc_topic_matrix.shape[0]}")
        sys.exit(1)

    # ==================== Compute document-topic assignments ====================
    log_time("Computing topic assignments from pre-computed matrix...")
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    max_scores = np.max(doc_topic_matrix, axis=1)

    log_time(f"Computed assignments for {len(dominant_topics)} documents")

    df_assignments = pd.DataFrame({
        'channel_id': df_messages['channel_id'].values,
        'dominant_topic': dominant_topics,
        'max_topic_score': max_scores
    })

    df_assignments.to_csv(assignments_path, index=False, compression='gzip')
    log_time(f"Saved document assignments to {assignments_path}")

    # ==================== Mark political documents ====================
    df_assignments['is_political'] = df_assignments['dominant_topic'].isin(politics_topics)
    political_docs = df_assignments['is_political'].sum()
    log_time(f"Political documents: {political_docs} ({100*political_docs/len(df_assignments):.1f}%)")

    # ==================== Compute per-channel statistics ====================
    log_time("Computing per-channel statistics...")
    channel_stats = df_assignments.groupby('channel_id').agg(
        total_messages=('is_political', 'count'),
        political_messages=('is_political', 'sum'),
        avg_topic_confidence=('max_topic_score', 'mean')
    ).reset_index()

    channel_stats['political_ratio'] = (
        channel_stats['political_messages'] / channel_stats['total_messages']
    )

    channel_stats = channel_stats.sort_values('political_ratio', ascending=False)
    channel_stats.to_csv(channel_stats_path, index=False)
    log_time(f"Saved channel stats to {channel_stats_path}")

    # ==================== Identify political channels ====================
    political_channels = channel_stats[
        channel_stats['political_ratio'] >= threshold
    ]['channel_id'].tolist()

    non_political_channels = channel_stats[
        channel_stats['political_ratio'] < threshold
    ]['channel_id'].tolist()

    log_time(f"Channels meeting threshold (>={threshold*100:.0f}%): {len(political_channels)}")
    log_time(f"Channels below threshold (<{threshold*100:.0f}%): {len(non_political_channels)}")

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

    # ==================== Print summary ====================
    log_time("=" * 60)
    log_time("SUMMARY STATISTICS:")
    log_time(f"  Total channels: {len(channel_stats)}")
    log_time(f"  Political (>={threshold*100:.0f}%): {len(political_channels)}")
    log_time(f"  Non-political (<{threshold*100:.0f}%): {len(non_political_channels)}")
    log_time(f"  Mean political ratio: {channel_stats['political_ratio'].mean():.2%}")
    log_time(f"  Median political ratio: {channel_stats['political_ratio'].median():.2%}")
    log_time(f"  Std political ratio: {channel_stats['political_ratio'].std():.2%}")

    log_time("\nDistribution of political ratios:")
    log_time(f"  0-20%: {len(channel_stats[channel_stats['political_ratio'] < 0.2])} channels")
    log_time(f"  20-40%: {len(channel_stats[(channel_stats['political_ratio'] >= 0.2) & (channel_stats['political_ratio'] < 0.4)])} channels")
    log_time(f"  40-60%: {len(channel_stats[(channel_stats['political_ratio'] >= 0.4) & (channel_stats['political_ratio'] < 0.6)])} channels")
    log_time(f"  60-80%: {len(channel_stats[(channel_stats['political_ratio'] >= 0.6) & (channel_stats['political_ratio'] < 0.8)])} channels")
    log_time(f"  80-100%: {len(channel_stats[channel_stats['political_ratio'] >= 0.8])} channels")

    log_time("\nTop 10 most political channels:")
    top10 = channel_stats.head(10)
    for _, row in top10.iterrows():
        log_time(f"  {row['channel_id']}: {row['political_ratio']:.1%} ({int(row['political_messages'])}/{int(row['total_messages'])} messages)")

    log_time("\nTop 10 least political channels:")
    bottom10 = channel_stats.tail(10)
    for _, row in bottom10.iterrows():
        log_time(f"  {row['channel_id']}: {row['political_ratio']:.1%} ({int(row['political_messages'])}/{int(row['total_messages'])} messages)")

    # ==================== Final timing ====================
    total_time = time.perf_counter() - START_TIME
    log_time(f"\nCOMPLETED in {total_time:.2f}s")

    with open(f"{channel_analysis_dir}/step5_completed.txt", "w") as f:
        f.write(f"Politics ratio computation completed in {total_time:.2f}s\n")
        f.write(f"Threshold: {threshold}\n")
        f.write(f"Total channels: {len(channel_stats)}\n")
        f.write(f"Political channels: {len(political_channels)}\n")
        f.write(f"Non-political channels: {len(non_political_channels)}\n")
        f.write(f"Mean political ratio: {channel_stats['political_ratio'].mean():.2%}\n")
        f.write(f"Median political ratio: {channel_stats['political_ratio'].median():.2%}\n")

if __name__ == "__main__":
    main()
