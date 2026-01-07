#!/usr/bin/env python3
"""
Step 5: Compute gaming ratio per channel.
A channel is "gaming" if gaming_ratio >= threshold.
"""

import os
import json
import argparse
import numpy as np
import pandas as pd

def compute_gaming_ratio(level, base_dir, threshold):
    """Compute gaming message ratio for each channel."""
    
    print(f"\n{'='*60}")
    print(f"STEP 5: GAMING RATIO - LEVEL {level}")
    print(f"Threshold: {threshold*100:.0f}%")
    print(f"{'='*60}")
    
    level_dir = f"{base_dir}/level_{level}"
    output_dir = f"{level_dir}/channel_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Load messages
    messages_file = f"{level_dir}/preprocessing/messages_english_clean.tsv.gz"
    if not os.path.exists(messages_file):
        print(f"[ERROR] Messages file not found")
        return
    
    df_messages = pd.read_csv(messages_file, sep='\t', compression='gzip')
    print(f"Total messages: {len(df_messages)}")
    print(f"Columns: {list(df_messages.columns)}")
    
    # Load doc-topic matrix
    matrix_file = f"{level_dir}/lda/doc_topic_matrix_level_{level}.npy"
    if not os.path.exists(matrix_file):
        print(f"[ERROR] Doc-topic matrix not found")
        return
    
    doc_topic_matrix = np.load(matrix_file)
    print(f"Doc-topic matrix shape: {doc_topic_matrix.shape}")
    
    # Load gaming topics
    gaming_file = f"{level_dir}/classification/gaming_topics.json"
    if not os.path.exists(gaming_file):
        # Try politics_topics.json for compatibility
        gaming_file = f"{level_dir}/classification/politics_topics.json"
    
    with open(gaming_file, 'r') as f:
        gaming_data = json.load(f)
    
    # Get gaming topic IDs
    gaming_topics = set(gaming_data.get('gaming_topics', gaming_data.get('politics_topics', [])))
    print(f"Gaming topics: {len(gaming_topics)}")
    
    # Get dominant topic for each message
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    
    # Check if message is gaming
    is_gaming = np.array([t in gaming_topics for t in dominant_topics])
    
    df_messages['is_gaming'] = is_gaming
    
    # Aggregate by channel - use 'text_lda' instead of 'message'
    channel_stats = df_messages.groupby('channel_id').agg(
        total_messages=('text_lda', 'count'),
        gaming_messages=('is_gaming', 'sum')
    ).reset_index()
    
    channel_stats['gaming_ratio'] = channel_stats['gaming_messages'] / channel_stats['total_messages']
    channel_stats['is_gaming_channel'] = channel_stats['gaming_ratio'] >= threshold
    
    # Summary
    gaming_channels = channel_stats[channel_stats['is_gaming_channel']]
    non_gaming_channels = channel_stats[~channel_stats['is_gaming_channel']]
    
    print(f"\nResults:")
    print(f"  Total channels: {len(channel_stats)}")
    print(f"  Gaming channels (>={threshold*100:.0f}%): {len(gaming_channels)}")
    print(f"  Non-gaming channels: {len(non_gaming_channels)}")
    print(f"  Mean gaming ratio: {channel_stats['gaming_ratio'].mean()*100:.1f}%")
    
    # Save results
    channel_stats.to_csv(f"{output_dir}/channel_stats.csv", index=False)
    
    results = {
        'threshold': threshold,
        'total_channels': len(channel_stats),
        'gaming_channels': len(gaming_channels),
        'non_gaming_channels': len(non_gaming_channels),
        'political_channel_ids': gaming_channels['channel_id'].tolist(),  # For compatibility
        'gaming_channel_ids': gaming_channels['channel_id'].tolist(),
        'mean_gaming_ratio': float(channel_stats['gaming_ratio'].mean()),
        'total_messages': int(channel_stats['total_messages'].sum()),
        'total_gaming_messages': int(channel_stats['gaming_messages'].sum()),
    }
    
    # Save as political_channels.json for compatibility with step6
    with open(f"{output_dir}/political_channels.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    with open(f"{output_dir}/gaming_channels.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Saved: {output_dir}/gaming_channels.json")
    
    with open(f"{output_dir}/step5_completed.txt", 'w') as f:
        f.write(f"Completed\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, required=True)
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()
    
    compute_gaming_ratio(args.level, args.base_dir, args.threshold)

if __name__ == "__main__":
    main()