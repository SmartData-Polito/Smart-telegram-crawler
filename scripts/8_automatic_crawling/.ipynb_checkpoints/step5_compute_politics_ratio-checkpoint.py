#!/usr/bin/env python3
"""
STEP 5: Compute political message percentage per channel.
Usage: python step5_compute_politics_ratio.py --level 0 --threshold 0.4
       python step5_compute_politics_ratio.py --level 0 --threshold 0.4 --base-dir ../../results/experiments/peak_jul_aug
"""

import os
import sys
import time
import argparse
import json
import numpy as np
import pandas as pd

# ======================== TIMING ========================
START_TIME = time.perf_counter()
STEP_TIMES = {}

def log_time(msg: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.2f}s] {msg}")

def start_timer(name: str) -> float:
    return time.perf_counter()

def end_timer(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    return elapsed

# ======================== CONFIG ========================
DEFAULT_THRESHOLD = 0.40

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    threshold = args.threshold
    
    log_time(f"Computing political ratios for level {level} (threshold={threshold})")
    log_time(f"  Base dir: {base_dir}")
    
    # Paths
    level_dir = f"{base_dir}/level_{level}"
    lda_dir = f"{level_dir}/lda"
    preprocess_dir = f"{level_dir}/preprocessing"
    classification_dir = f"{level_dir}/classification"
    channel_analysis_dir = f"{level_dir}/channel_analysis"
    os.makedirs(channel_analysis_dir, exist_ok=True)
    
    # Load nodes file for tracking
    t_start = start_timer("load_nodes")
    nodes_file = f"{level_dir}/nodes_level_{level}.csv.gz"
    all_input_channels = set()
    if os.path.exists(nodes_file):
        df_nodes = pd.read_csv(nodes_file, compression='gzip')
        all_input_channels = set(df_nodes['type_and_id'].tolist())
        log_time(f"Loaded {len(all_input_channels)} channels from nodes file")
    end_timer("load_nodes", t_start)
    
    # Load channels tracking from step1
    t_start = start_timer("load_tracking")
    channels_tracking_path = f"{preprocess_dir}/channels_tracking.json"
    channels_tracking = None
    if os.path.exists(channels_tracking_path):
        with open(channels_tracking_path, 'r') as f:
            channels_tracking = json.load(f)
        log_time(f"Loaded channels tracking from step1")
    end_timer("load_tracking", t_start)
    
    # Load politics topics
    t_start = start_timer("load_politics_topics")
    politics_topics_path = f"{classification_dir}/politics_topics.json"
    if not os.path.exists(politics_topics_path):
        log_time(f"ERROR: Politics topics not found: {politics_topics_path}")
        sys.exit(1)
    
    with open(politics_topics_path, "r") as f:
        politics_data = json.load(f)
    politics_topics = set(politics_data["politics_topics"])
    log_time(f"Loaded {len(politics_topics)} politics topics")
    end_timer("load_politics_topics", t_start)
    
    # Load doc_topic_matrix
    t_start = start_timer("load_doc_topic_matrix")
    doc_topic_matrix_path = f"{lda_dir}/doc_topic_matrix_level_{level}.npy"
    if not os.path.exists(doc_topic_matrix_path):
        log_time(f"ERROR: doc_topic_matrix not found: {doc_topic_matrix_path}")
        sys.exit(1)
    
    log_time("Loading pre-computed doc_topic_matrix...")
    doc_topic_matrix = np.load(doc_topic_matrix_path)
    log_time(f"doc_topic_matrix shape: {doc_topic_matrix.shape}")
    end_timer("load_doc_topic_matrix", t_start)
    
    # Load messages
    t_start = start_timer("load_messages")
    messages_path = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    if not os.path.exists(messages_path):
        log_time(f"ERROR: Messages file not found: {messages_path}")
        sys.exit(1)
    
    log_time("Loading messages...")
    df_messages = pd.read_csv(messages_path, sep='\t', compression='gzip',
                               usecols=['channel_id', 'text_lda'])
    df_messages = df_messages[df_messages['text_lda'].astype(str).str.strip() != '']
    log_time(f"Loaded {len(df_messages)} messages")
    end_timer("load_messages", t_start)
    
    # Verify dimensions
    if len(df_messages) != doc_topic_matrix.shape[0]:
        log_time(f"ERROR: Mismatch! Messages: {len(df_messages)}, matrix rows: {doc_topic_matrix.shape[0]}")
        sys.exit(1)
    
    # Compute topic assignments
    t_start = start_timer("compute_assignments")
    log_time("Computing topic assignments...")
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    max_scores = np.max(doc_topic_matrix, axis=1)
    log_time(f"Computed assignments for {len(dominant_topics)} documents")
    end_timer("compute_assignments", t_start)
    
    # Save assignments
    t_start = start_timer("save_assignments")
    df_assignments = pd.DataFrame({
        'channel_id': df_messages['channel_id'].values,
        'dominant_topic': dominant_topics,
        'max_topic_score': max_scores
    })
    
    assignments_path = f"{channel_analysis_dir}/doc_topic_assignments.csv.gz"
    df_assignments.to_csv(assignments_path, index=False, compression='gzip')
    log_time(f"Saved document assignments to {assignments_path}")
    end_timer("save_assignments", t_start)
    
    # Mark political documents
    t_start = start_timer("mark_political")
    df_assignments['is_political'] = df_assignments['dominant_topic'].isin(politics_topics)
    political_docs = df_assignments['is_political'].sum()
    log_time(f"Political documents: {political_docs} ({100*political_docs/len(df_assignments):.1f}%)")
    end_timer("mark_political", t_start)
    
    # Compute per-channel statistics
    t_start = start_timer("compute_channel_stats")
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
    end_timer("compute_channel_stats", t_start)
    
    # Save channel stats
    t_start = start_timer("save_channel_stats")
    channel_stats_path = f"{channel_analysis_dir}/channel_politics_stats.csv"
    channel_stats.to_csv(channel_stats_path, index=False)
    log_time(f"Saved channel stats to {channel_stats_path}")
    end_timer("save_channel_stats", t_start)
    
    # Identify political channels
    t_start = start_timer("identify_political_channels")
    political_channels = channel_stats[
        channel_stats['political_ratio'] >= threshold
    ]['channel_id'].tolist()
    
    non_political_channels = channel_stats[
        channel_stats['political_ratio'] < threshold
    ]['channel_id'].tolist()
    
    # Track missing channels
    classified_channels = set(channel_stats['channel_id'].tolist())
    missing_channels = sorted(list(all_input_channels - classified_channels))
    end_timer("identify_political_channels", t_start)
    
    # Build output data
    t_start = start_timer("save_results")
    output_data = {
        "level": level,
        "threshold": threshold,
        "summary": {
            "input_channels": len(all_input_channels),
            "classified_channels": len(classified_channels),
            "missing_channels": len(missing_channels),
            "political_channels": len(political_channels),
            "non_political_channels": len(non_political_channels)
        },
        "details": {
            "political_channels": political_channels,
            "non_political_channels": non_political_channels,
            "missing_channels": missing_channels
        }
    }
    
    if channels_tracking:
        output_data["missing_channels_breakdown"] = channels_tracking.get("summary", {})
        output_data["date_filter"] = {
            "start_date": channels_tracking.get("start_date"),
            "end_date": channels_tracking.get("end_date")
        }
    
    political_channels_path = f"{channel_analysis_dir}/political_channels.json"
    with open(political_channels_path, "w") as f:
        json.dump(output_data, f, indent=2)
    log_time(f"Saved political channels to {political_channels_path}")
    end_timer("save_results", t_start)
    
    # Print summary
    log_time(f"\n{'='*60}")
    log_time("SUMMARY:")
    log_time(f"  Threshold: {threshold*100:.0f}%")
    log_time(f"  Input channels: {len(all_input_channels)}")
    log_time(f"  Classified: {len(classified_channels)}")
    log_time(f"  Missing: {len(missing_channels)}")
    log_time(f"  Political (>={threshold*100:.0f}%): {len(political_channels)}")
    log_time(f"  Non-political (<{threshold*100:.0f}%): {len(non_political_channels)}")
    log_time(f"  Mean political ratio: {channel_stats['political_ratio'].mean():.2%}")
    log_time(f"  Median political ratio: {channel_stats['political_ratio'].median():.2%}")
    log_time(f"{'='*60}")
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"\nCOMPLETED in {total_time:.2f}s")
    
    with open(f"{channel_analysis_dir}/step5_completed.txt", 'w') as f:
        f.write(f"Step 5: Politics Ratio Computation\n")
        f.write(f"Level: {level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Threshold: {threshold}\n")
        f.write(f"  Input channels: {len(all_input_channels)}\n")
        f.write(f"  Classified: {len(classified_channels)}\n")
        f.write(f"  Political: {len(political_channels)}\n")
        f.write(f"  Non-political: {len(non_political_channels)}\n")
        f.write(f"  Mean ratio: {channel_stats['political_ratio'].mean():.2%}\n")
        f.write(f"  Median ratio: {channel_stats['political_ratio'].median():.2%}\n")

if __name__ == "__main__":
    main()