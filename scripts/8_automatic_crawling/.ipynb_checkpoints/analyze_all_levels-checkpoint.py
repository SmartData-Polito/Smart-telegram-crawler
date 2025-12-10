#!/usr/bin/env python3
"""
Complete analysis of all levels in the pipeline.
Usage: python analyze_all_levels.py --threshold 0.4
       python analyze_all_levels.py --threshold 0.4 --level 0
       python analyze_all_levels.py --threshold 0.4 --level non_visited
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
    print(f"ANALYZING LEVEL: {level}")
    print(f"{'='*70}")
    
    # Check if level has been processed
    if not os.path.exists(f"{preprocess_dir}/messages_english_clean.tsv.gz"):
        print(f"  [SKIP] No preprocessed messages found")
        return None
    
    if not os.path.exists(f"{lda_dir}/doc_topic_matrix_level_{level}.npy"):
        print(f"  [SKIP] No doc_topic_matrix found")
        return None
    
    # Load doc_topic_matrix
    print("  Loading doc_topic_matrix...")
    doc_topic_matrix = np.load(f"{lda_dir}/doc_topic_matrix_level_{level}.npy")
    n_docs, n_topics = doc_topic_matrix.shape
    print(f"    Shape: {n_docs} documents x {n_topics} topics")
    
    # Load messages
    print("  Loading messages...")
    df_messages = pd.read_csv(f"{preprocess_dir}/messages_english_clean.tsv.gz",
                               sep='\t', compression='gzip')
    print(f"    Loaded {len(df_messages)} messages")
    
    # Load politics topics
    politics_topics = set()
    politics_path = f"{classification_dir}/politics_topics.json"
    if os.path.exists(politics_path):
        with open(politics_path, 'r') as f:
            politics_data = json.load(f)
            politics_topics = set(politics_data.get("politics_topics", []))
    print(f"    Political topics: {len(politics_topics)} / {n_topics}")
    
    # Load topic keywords
    topics_keywords = {}
    topics_path = f"{level_dir}/topics/topics_for_classification.json"
    if os.path.exists(topics_path):
        with open(topics_path, 'r') as f:
            topics_data = json.load(f)
        
        # Handle different formats
        if isinstance(topics_data, dict) and "topics" in topics_data:
            # Format: {"level": "0", "num_topics": 94, "topics": [{"topic_id": 0, "keywords": [...]}, ...]}
            for t in topics_data["topics"]:
                if isinstance(t, dict):
                    topic_id = t.get('topic_id', len(topics_keywords))
                    topics_keywords[topic_id] = t.get('keywords', [])[:10]
        elif isinstance(topics_data, list):
            # Format: [{"topic_id": 0, "keywords": [...]}, ...]
            for t in topics_data:
                if isinstance(t, dict):
                    topic_id = t.get('topic_id', len(topics_keywords))
                    topics_keywords[topic_id] = t.get('keywords', [])[:10]
        elif isinstance(topics_data, dict):
            # Format: {"0": {"keywords": [...]}, ...}
            for k, v in topics_data.items():
                try:
                    topic_id = int(k)
                except ValueError:
                    continue
                
                if isinstance(v, dict):
                    topics_keywords[topic_id] = v.get('keywords', [])[:10]
                elif isinstance(v, list):
                    topics_keywords[topic_id] = v[:10]
                elif isinstance(v, str):
                    topics_keywords[topic_id] = [w.strip() for w in v.split(',')][:10]
    
    print(f"    Loaded keywords for {len(topics_keywords)} topics")
    
    # Compute dominant topic and political status for each message
    dominant_topics = np.argmax(doc_topic_matrix, axis=1)
    df_messages['dominant_topic'] = dominant_topics
    df_messages['is_political_message'] = df_messages['dominant_topic'].isin(politics_topics)
    
    # ==================== LEVEL STATISTICS ====================
    total_messages = len(df_messages)
    political_messages = int(df_messages['is_political_message'].sum())
    non_political_messages = total_messages - political_messages
    
    print(f"\n  LEVEL STATISTICS:")
    print(f"    Total messages: {total_messages:,}")
    print(f"    Political messages: {political_messages:,} ({100*political_messages/total_messages:.1f}%)")
    print(f"    Non-political messages: {non_political_messages:,} ({100*non_political_messages/total_messages:.1f}%)")
    
    # ==================== GROUP/CHANNEL STATISTICS ====================
    channel_stats = df_messages.groupby('channel_id').agg(
        total_messages=('is_political_message', 'count'),
        political_messages=('is_political_message', 'sum')
    ).reset_index()
    
    channel_stats['non_political_messages'] = channel_stats['total_messages'] - channel_stats['political_messages']
    channel_stats['political_ratio'] = channel_stats['political_messages'] / channel_stats['total_messages']
    channel_stats['is_political_channel'] = channel_stats['political_ratio'] >= threshold
    
    total_channels = len(channel_stats)
    political_channels = int(channel_stats['is_political_channel'].sum())
    non_political_channels = total_channels - political_channels
    
    print(f"\n  CHANNEL STATISTICS (threshold={threshold*100:.0f}%):")
    print(f"    Total channels: {total_channels:,}")
    print(f"    Political channels: {political_channels:,} ({100*political_channels/total_channels:.1f}%)")
    print(f"    Non-political channels: {non_political_channels:,} ({100*non_political_channels/total_channels:.1f}%)")
    
    # ==================== POLITICAL CHANNELS DETAILS ====================
    political_channel_details = []
    df_political_channels = channel_stats[channel_stats['is_political_channel']]
    
    for _, row in df_political_channels.iterrows():
        political_channel_details.append({
            "channel_id": row['channel_id'],
            "total_messages": int(row['total_messages']),
            "political_messages": int(row['political_messages']),
            "non_political_messages": int(row['non_political_messages']),
            "political_ratio": round(float(row['political_ratio']), 4)
        })
    
    political_channel_details = sorted(political_channel_details, 
                                        key=lambda x: x['political_messages'], reverse=True)
    
    # ==================== NON-POLITICAL CHANNELS DETAILS ====================
    non_political_channel_details = []
    df_non_political_channels = channel_stats[~channel_stats['is_political_channel']]
    
    for _, row in df_non_political_channels.iterrows():
        non_political_channel_details.append({
            "channel_id": row['channel_id'],
            "total_messages": int(row['total_messages']),
            "political_messages": int(row['political_messages']),
            "non_political_messages": int(row['non_political_messages']),
            "political_ratio": round(float(row['political_ratio']), 4)
        })
    
    non_political_channel_details = sorted(non_political_channel_details,
                                            key=lambda x: x['total_messages'], reverse=True)
    
    # ==================== TOPIC STATISTICS ====================
    topic_stats = []
    
    for topic_id in range(n_topics):
        topic_mask = df_messages['dominant_topic'] == topic_id
        topic_messages = int(topic_mask.sum())
        
        is_political_topic = topic_id in politics_topics
        keywords = topics_keywords.get(topic_id, [])
        
        topic_stats.append({
            "topic_id": topic_id,
            "is_political_topic": is_political_topic,
            "keywords": keywords,
            "total_messages": topic_messages,
            "messages_classification": "political" if is_political_topic else "non_political"
        })
    
    topic_stats = sorted(topic_stats, key=lambda x: x['total_messages'], reverse=True)
    
    print(f"\n  TOP 10 TOPICS:")
    for t in topic_stats[:10]:
        pol_marker = "[P]" if t['is_political_topic'] else "[ ]"
        kw_preview = ", ".join(t['keywords'][:5]) if t['keywords'] else "N/A"
        print(f"    Topic {t['topic_id']:3} {pol_marker}: {t['total_messages']:>6,} msgs - {kw_preview}")
    
    # ==================== CHANNEL-TOPIC DISTRIBUTION ====================
    channel_topic_distribution = []
    channel_topic_counts = df_messages.groupby(['channel_id', 'dominant_topic']).size().reset_index(name='count')
    
    for channel_id in channel_stats['channel_id'].unique():
        ch_topics = channel_topic_counts[channel_topic_counts['channel_id'] == channel_id]
        
        topic_dist = {}
        for _, row in ch_topics.iterrows():
            topic_id = int(row['dominant_topic'])
            count = int(row['count'])
            topic_dist[topic_id] = count
        
        ch_row = channel_stats[channel_stats['channel_id'] == channel_id].iloc[0]
        
        channel_topic_distribution.append({
            "channel_id": channel_id,
            "is_political_channel": bool(ch_row['is_political_channel']),
            "total_messages": int(ch_row['total_messages']),
            "political_ratio": round(float(ch_row['political_ratio']), 4),
            "topic_distribution": topic_dist
        })
    
    channel_topic_distribution = sorted(channel_topic_distribution,
                                         key=lambda x: x['total_messages'], reverse=True)
    
    # ==================== BUILD RESULT ====================
    result = {
        "level": level,
        "is_non_visited": level == "non_visited",
        "threshold": threshold,
        
        "level_summary": {
            "total_messages": int(total_messages),
            "political_messages": int(political_messages),
            "non_political_messages": int(non_political_messages),
            "political_message_ratio": round(political_messages / total_messages, 4) if total_messages > 0 else 0,
            
            "total_channels": int(total_channels),
            "political_channels": int(political_channels),
            "non_political_channels": int(non_political_channels),
            "political_channel_ratio": round(political_channels / total_channels, 4) if total_channels > 0 else 0,
            
            "total_topics": n_topics,
            "political_topics": len(politics_topics),
            "non_political_topics": n_topics - len(politics_topics)
        },
        
        "political_channels": political_channel_details,
        "non_political_channels": non_political_channel_details,
        "topics": topic_stats,
        "channel_topic_distribution": channel_topic_distribution
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, default=0.4, 
                        help="Threshold for political channel classification (default: 0.4)")
    parser.add_argument("--level", type=str, default=None,
                        help="Analyze specific level only (e.g., 0, 1, non_visited)")
    parser.add_argument("--output-dir", type=str, default="../../results/levels_automatic/analysis",
                        help="Output directory for analysis results")
    args = parser.parse_args()
    
    base_dir = "../../results/levels_automatic"
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"=" * 70)
    print(f"COMPLETE PIPELINE ANALYSIS")
    print(f"Threshold: {args.threshold*100:.0f}%")
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
    
    # Analyze each level
    all_results = {}
    summary_rows = []
    
    for level in levels:
        result = analyze_level(level, args.threshold, base_dir)
        
        if result is not None:
            all_results[level] = result
            
            summary_rows.append({
                "level": level,
                "is_non_visited": result["is_non_visited"],
                "total_messages": result["level_summary"]["total_messages"],
                "political_messages": result["level_summary"]["political_messages"],
                "non_political_messages": result["level_summary"]["non_political_messages"],
                "political_message_ratio": result["level_summary"]["political_message_ratio"],
                "total_channels": result["level_summary"]["total_channels"],
                "political_channels": result["level_summary"]["political_channels"],
                "non_political_channels": result["level_summary"]["non_political_channels"],
                "political_channel_ratio": result["level_summary"]["political_channel_ratio"],
                "total_topics": result["level_summary"]["total_topics"],
                "political_topics": result["level_summary"]["political_topics"]
            })
            
            # Save individual level analysis
            level_output = f"{output_dir}/level_{level}_analysis.json"
            with open(level_output, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"  Saved: {level_output}")
            
            # Save channel details CSV
            if result["political_channels"]:
                df_pol = pd.DataFrame(result["political_channels"])
                df_pol.to_csv(f"{output_dir}/level_{level}_political_channels.csv", index=False)
            
            if result["non_political_channels"]:
                df_non_pol = pd.DataFrame(result["non_political_channels"])
                df_non_pol.to_csv(f"{output_dir}/level_{level}_non_political_channels.csv", index=False)
            
            # Save topics CSV
            df_topics = pd.DataFrame(result["topics"])
            df_topics['keywords_str'] = df_topics['keywords'].apply(lambda x: ", ".join(x[:5]) if x else "")
            df_topics = df_topics[['topic_id', 'is_political_topic', 'total_messages', 'keywords_str']]
            df_topics.to_csv(f"{output_dir}/level_{level}_topics.csv", index=False)
            
            # Save channel-topic distribution
            channel_topic_rows = []
            for ch in result["channel_topic_distribution"]:
                for topic_id, count in ch["topic_distribution"].items():
                    channel_topic_rows.append({
                        "channel_id": ch["channel_id"],
                        "is_political_channel": ch["is_political_channel"],
                        "topic_id": topic_id,
                        "message_count": count
                    })
            
            if channel_topic_rows:
                df_ch_topic = pd.DataFrame(channel_topic_rows)
                df_ch_topic.to_csv(f"{output_dir}/level_{level}_channel_topic_matrix.csv", index=False)
    
    # Save overall summary
    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        summary_csv = f"{output_dir}/all_levels_summary.csv"
        df_summary.to_csv(summary_csv, index=False)
        print(f"\nSaved overall summary: {summary_csv}")
        
        complete_output = f"{output_dir}/complete_analysis.json"
        with open(complete_output, 'w') as f:
            json.dump(all_results, f, indent=2)
        print(f"Saved complete analysis: {complete_output}")
    
    # Print final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    if summary_rows:
        bfs_rows = [r for r in summary_rows if not r["is_non_visited"]]
        non_visited_rows = [r for r in summary_rows if r["is_non_visited"]]
        
        if bfs_rows:
            print("\nBFS STRATEGY LEVELS:")
            print(f"{'Level':<10} {'Messages':>12} {'Pol Msgs':>12} {'Pol %':>8} {'Channels':>10} {'Pol Ch':>10} {'Pol Ch %':>10}")
            print("-" * 80)
            
            total_msgs = 0
            total_pol_msgs = 0
            total_ch = 0
            total_pol_ch = 0
            
            for r in bfs_rows:
                print(f"{r['level']:<10} {r['total_messages']:>12,} {r['political_messages']:>12,} "
                      f"{r['political_message_ratio']*100:>7.1f}% {r['total_channels']:>10,} "
                      f"{r['political_channels']:>10,} {r['political_channel_ratio']*100:>9.1f}%")
                total_msgs += r['total_messages']
                total_pol_msgs += r['political_messages']
                total_ch += r['total_channels']
                total_pol_ch += r['political_channels']
            
            print("-" * 80)
            if total_msgs > 0 and total_ch > 0:
                print(f"{'TOTAL':<10} {total_msgs:>12,} {total_pol_msgs:>12,} "
                      f"{100*total_pol_msgs/total_msgs:>7.1f}% {total_ch:>10,} "
                      f"{total_pol_ch:>10,} {100*total_pol_ch/total_ch:>9.1f}%")
        
        if non_visited_rows:
            print("\nNON-VISITED (channels NOT explored by BFS strategy):")
            print(f"{'Level':<15} {'Messages':>12} {'Pol Msgs':>12} {'Pol %':>8} {'Channels':>10} {'Pol Ch':>10} {'Pol Ch %':>10}")
            print("-" * 85)
            
            for r in non_visited_rows:
                print(f"{r['level']:<15} {r['total_messages']:>12,} {r['political_messages']:>12,} "
                      f"{r['political_message_ratio']*100:>7.1f}% {r['total_channels']:>10,} "
                      f"{r['political_channels']:>10,} {r['political_channel_ratio']*100:>9.1f}%")
        
        # Comparison
        if bfs_rows and non_visited_rows:
            print("\n" + "="*70)
            print("STRATEGY COMPARISON")
            print("="*70)
            
            bfs_total_msgs = sum(r['total_messages'] for r in bfs_rows)
            bfs_pol_msgs = sum(r['political_messages'] for r in bfs_rows)
            bfs_total_ch = sum(r['total_channels'] for r in bfs_rows)
            bfs_pol_ch = sum(r['political_channels'] for r in bfs_rows)
            
            nv = non_visited_rows[0]
            
            print(f"\n{'Metric':<30} {'BFS Strategy':>20} {'Non-Visited':>20} {'Difference':>15}")
            print("-" * 90)
            print(f"{'Total Messages':<30} {bfs_total_msgs:>20,} {nv['total_messages']:>20,}")
            print(f"{'Political Messages':<30} {bfs_pol_msgs:>20,} {nv['political_messages']:>20,}")
            
            bfs_msg_ratio = bfs_pol_msgs/bfs_total_msgs if bfs_total_msgs > 0 else 0
            nv_msg_ratio = nv['political_message_ratio']
            print(f"{'Political Message Ratio':<30} {100*bfs_msg_ratio:>19.1f}% {nv_msg_ratio*100:>19.1f}% "
                  f"{100*bfs_msg_ratio - nv_msg_ratio*100:>+14.1f}%")
            
            print(f"{'Total Channels':<30} {bfs_total_ch:>20,} {nv['total_channels']:>20,}")
            print(f"{'Political Channels':<30} {bfs_pol_ch:>20,} {nv['political_channels']:>20,}")
            
            bfs_ch_ratio = bfs_pol_ch/bfs_total_ch if bfs_total_ch > 0 else 0
            nv_ch_ratio = nv['political_channel_ratio']
            print(f"{'Political Channel Ratio':<30} {100*bfs_ch_ratio:>19.1f}% {nv_ch_ratio*100:>19.1f}% "
                  f"{100*bfs_ch_ratio - nv_ch_ratio*100:>+14.1f}%")
            
            print("\nINTERPRETATION:")
            
            if bfs_ch_ratio > nv_ch_ratio and nv_ch_ratio > 0:
                improvement = (bfs_ch_ratio - nv_ch_ratio) / nv_ch_ratio * 100
                print(f"  ✓ BFS strategy found {improvement:.1f}% MORE political channels than random selection would")
                print(f"  ✓ Strategy is EFFECTIVE at finding political content")
            elif bfs_ch_ratio > nv_ch_ratio:
                print(f"  ✓ BFS strategy has higher political channel ratio")
                print(f"  ✓ Strategy is EFFECTIVE at finding political content")
            else:
                print(f"  ✗ BFS strategy did NOT improve political channel discovery")
                print(f"  ✗ Non-visited channels have similar or higher political ratio")

if __name__ == "__main__":
    main()