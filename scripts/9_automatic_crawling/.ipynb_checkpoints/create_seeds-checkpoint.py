#!/usr/bin/env python3
"""
Create seeds for TGDataset experiments.
This script is separate from the pipeline and should not count in total time.

Creates two types of seed sets:
- PURE: 100% gaming channels (~5% of total dataset)
- MIXED: 50% gaming + 50% non-gaming (~5% of total dataset)

Uses stratified sampling based on forward_count (10 quantiles, proportional).
"""

import os
import sys
import json
import tarfile
import argparse
import time
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

from openai import OpenAI

# ======================== CONFIGURATION ========================
TGDATASET_DIR = "../../material"
LABELED_DATA_DIR = f"{TGDATASET_DIR}/TGDataset/labeled_data"
OUTPUT_DIR = "../../results/experiments_tgdataset"

TARGET_SEED_PERCENTAGE = 0.05  # 5% of total dataset
TOTAL_CHANNELS = 120979  # TGDataset total
TARGET_SEEDS = int(TOTAL_CHANNELS * TARGET_SEED_PERCENTAGE)  # ~6049

N_QUANTILES = 10
MODEL_NAME = "gpt-5-nano"

# Default gaming topics (fallback)
DEFAULT_GAMING_TOPICS = ['Videogame modding']

# ======================== HELPER FUNCTIONS ========================
def load_api_key():
    """Load OpenAI API key."""
    api_key_paths = [
        "../openai_secrets.json",
        "openai_secrets.json",
        os.path.expanduser("~/openai_secrets.json")
    ]
    
    api_key = None
    for path in api_key_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                secrets = json.load(f)
                api_key = secrets.get('api_key') or secrets.get('OPENAI_API_KEY')
                break
    
    if not api_key:
        api_key = os.environ.get('OPENAI_API_KEY')
    
    return api_key

def classify_gaming_topics(topics):
    """
    Use GPT to classify which topics are gaming-related.
    Be strict: only clearly gaming-related topics.
    """
    print(f"\nClassifying {len(topics)} topics with GPT ({MODEL_NAME})...")
    
    api_key = load_api_key()
    if not api_key:
        print("[WARN] OpenAI API key not found, using defaults")
        return DEFAULT_GAMING_TOPICS
    
    client = OpenAI(api_key=api_key)
    
    prompt = f"""You are classifying Telegram channel topics. Be STRICT about what counts as gaming.

Topics to classify:
{chr(10).join(f'- {t}' for t in topics)}

For each topic, decide if it's GAMING-related (video games, game mods, esports, gaming communities).

Be STRICT:
- "Videogame modding" → YES (clearly gaming)
- "Entertainment" → NO (too broad, not specifically gaming)
- "Software" → NO (too broad)
- "Crypto" → NO (not gaming)

Return ONLY a JSON list of gaming topic names, nothing else.
Example: ["Videogame modding", "Esports"]
If none are gaming, return: []
"""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=600
        )
        
        result = response.choices[0].message.content.strip()
        # Clean up result if needed
        if result.startswith("```"):
            result = result.split("```")[1]
            if result.startswith("json"):
                result = result[4:]
        
        gaming_topics = json.loads(result)
        print(f"  GPT identified gaming topics: {gaming_topics}")
        return gaming_topics
    except Exception as e:
        print(f"[WARN] GPT classification failed: {e}")
        print(f"[WARN] Using default: {DEFAULT_GAMING_TOPICS}")
        return DEFAULT_GAMING_TOPICS

def load_topic_mapping():
    """Load channel to topic mapping."""
    topic_file = f"{LABELED_DATA_DIR}/ch_to_topic_mapping.csv"
    df = pd.read_csv(topic_file)
    print(f"Loaded {len(df)} labeled channels")
    print(f"\nAvailable topics:")
    for topic, count in df['topic'].value_counts().items():
        print(f"  {topic}: {count}")
    return df

def load_or_build_channel_mapping():
    """Load or build mapping from channel_id to tar file and JSON path."""
    mapping_file = f"{TGDATASET_DIR}/channel_file_mapping.json"
    
    if os.path.exists(mapping_file):
        print(f"Loading existing channel mapping...")
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    print("Building channel to file mapping (this takes ~30 min first time)...")
    mapping = {}
    
    for i in range(1, 5):
        tar_path = f"{TGDATASET_DIR}/TGDataset_{i}.tar.gz"
        if not os.path.exists(tar_path):
            print(f"  [SKIP] {tar_path} not found")
            continue
            
        print(f"  Scanning TGDataset_{i}.tar.gz...")
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            for member in tqdm(tar.getmembers()):
                if member.name.endswith('.json'):
                    try:
                        f = tar.extractfile(member)
                        if f:
                            data = json.load(f)
                            for ch_id in data.keys():
                                mapping[str(ch_id)] = {
                                    'tar': f'TGDataset_{i}.tar.gz',
                                    'file': member.name
                                }
                    except:
                        pass
    
    with open(mapping_file, 'w') as f:
        json.dump(mapping, f)
    
    print(f"  Saved mapping: {len(mapping)} channels")
    return mapping

def compute_forward_counts(channel_ids, channel_mapping):
    """
    Compute forward_count for each channel.
    forward_count = number of messages with forwarded_from_id in that channel.
    """
    print(f"\nComputing forward counts for {len(channel_ids)} channels...")
    
    forward_counts = {}
    
    # Group channels by tar file for efficiency
    by_tar = defaultdict(list)
    for ch_id in channel_ids:
        ch_id_str = str(ch_id)
        if ch_id_str in channel_mapping:
            info = channel_mapping[ch_id_str]
            by_tar[info['tar']].append((ch_id_str, info['file']))
    
    for tar_name, items in by_tar.items():
        tar_path = f"{TGDATASET_DIR}/{tar_name}"
        print(f"  Processing {tar_name}...")
        
        # Group by file
        by_file = defaultdict(list)
        for ch_id, file_path in items:
            by_file[file_path].append(ch_id)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            for file_path, ch_ids_in_file in tqdm(by_file.items()):
                try:
                    f = tar.extractfile(file_path)
                    if f:
                        data = json.load(f)
                        for ch_id in ch_ids_in_file:
                            if ch_id in data:
                                channel_data = data[ch_id]
                                text_messages = channel_data.get('text_messages', {})
                                
                                # Count messages with forwarded_from_id
                                fwd_count = sum(
                                    1 for msg in text_messages.values()
                                    if msg.get('forwarded_from_id') is not None
                                )
                                forward_counts[int(ch_id)] = fwd_count
                except Exception as e:
                    pass
    
    # Channels not found get 0
    for ch_id in channel_ids:
        if ch_id not in forward_counts:
            forward_counts[ch_id] = 0
    
    return forward_counts

def stratified_sample(df, n_samples, forward_count_col='forward_count'):
    """
    Stratified sampling based on forward_count with 10 quantiles.
    Proportional sampling from each quantile.
    """
    print(f"\nStratified sampling: {n_samples} from {len(df)} channels")
    
    # Create quantile labels
    df = df.copy()
    df['quantile'] = pd.qcut(
        df[forward_count_col].rank(method='first'), 
        q=N_QUANTILES, 
        labels=range(N_QUANTILES)
    )
    
    # Count per quantile
    quantile_counts = df['quantile'].value_counts().sort_index()
    print(f"\nQuantile distribution:")
    for q, count in quantile_counts.items():
        pct = count / len(df) * 100
        q_df = df[df['quantile'] == q]
        min_fwd = q_df[forward_count_col].min()
        max_fwd = q_df[forward_count_col].max()
        print(f"  Q{q}: {count} channels ({pct:.1f}%), forward_count: {min_fwd}-{max_fwd}")
    
    # If we want more samples than available, take all
    if n_samples >= len(df):
        print(f"\n[INFO] Requested {n_samples} but only {len(df)} available. Taking all.")
        return df
    
    # Proportional sampling
    sampled = df.groupby('quantile', group_keys=False).apply(
        lambda x: x.sample(frac=n_samples/len(df), random_state=42)
    )
    
    # Adjust if we're short
    while len(sampled) < n_samples:
        remaining = df[~df.index.isin(sampled.index)]
        if len(remaining) == 0:
            break
        sampled = pd.concat([sampled, remaining.sample(1, random_state=42)])
    
    print(f"\nSampled {len(sampled)} channels")
    print(f"Sample quantile distribution:")
    for q, count in sampled['quantile'].value_counts().sort_index().items():
        pct = count / len(sampled) * 100
        print(f"  Q{q}: {count} ({pct:.1f}%)")
    
    return sampled

def create_seeds(experiment_type, threshold, gaming_channels_df, non_gaming_channels_df):
    """
    Create seed set for an experiment.
    
    experiment_type: 'pure' or 'mixed'
    threshold: 20, 40, 60, or 80
    """
    experiment_name = f"threshold_{threshold}_{experiment_type}"
    print(f"\n{'='*60}")
    print(f"CREATING SEEDS: {experiment_name}")
    print(f"{'='*60}")
    
    if experiment_type == 'pure':
        # 100% gaming, target ~5%
        n_seeds = min(TARGET_SEEDS, len(gaming_channels_df))
        
        if len(gaming_channels_df) <= TARGET_SEEDS:
            print(f"Taking all {len(gaming_channels_df)} gaming channels")
            seeds_df = gaming_channels_df.copy()
        else:
            print(f"Stratified sampling {n_seeds} from {len(gaming_channels_df)} gaming channels")
            seeds_df = stratified_sample(gaming_channels_df, n_seeds)
        
        seeds_df['seed_type'] = 'gaming'
        
    else:  # mixed
        # 50% gaming + 50% non-gaming, total ~5%
        n_gaming = TARGET_SEEDS // 2
        n_non_gaming = TARGET_SEEDS // 2
        
        # Gaming (stratified)
        if len(gaming_channels_df) <= n_gaming:
            print(f"Taking all {len(gaming_channels_df)} gaming channels")
            gaming_seeds = gaming_channels_df.copy()
        else:
            print(f"Stratified sampling {n_gaming} gaming channels")
            gaming_seeds = stratified_sample(gaming_channels_df, n_gaming)
        gaming_seeds['seed_type'] = 'gaming'
        
        # Non-gaming (random)
        if len(non_gaming_channels_df) <= n_non_gaming:
            print(f"Taking all {len(non_gaming_channels_df)} non-gaming channels")
            non_gaming_seeds = non_gaming_channels_df.copy()
        else:
            print(f"Random sampling {n_non_gaming} non-gaming channels")
            non_gaming_seeds = non_gaming_channels_df.sample(n=n_non_gaming, random_state=42)
        non_gaming_seeds['seed_type'] = 'non_gaming'
        
        seeds_df = pd.concat([gaming_seeds, non_gaming_seeds], ignore_index=True)
    
    # Create output directory
    output_dir = f"{OUTPUT_DIR}/{experiment_name}/level_0"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save nodes file (only channel_id - pipeline doesn't see labels)
    nodes_df = seeds_df[['channel_id']].copy()
    nodes_file = f"{output_dir}/nodes_level_0.csv.gz"
    nodes_df.to_csv(nodes_file, index=False, compression='gzip')
    print(f"\nSaved: {nodes_file}")
    print(f"  Total seeds: {len(nodes_df)}")
    
    # Save detailed seed info (for reference, not used by pipeline)
    seed_info = {
        'experiment_name': experiment_name,
        'experiment_type': experiment_type,
        'threshold': threshold,
        'total_seeds': len(seeds_df),
        'gaming_seeds': int(len(seeds_df[seeds_df['seed_type'] == 'gaming'])),
        'non_gaming_seeds': int(len(seeds_df[seeds_df['seed_type'] == 'non_gaming'])) if 'seed_type' in seeds_df.columns else 0,
        'target_percentage': TARGET_SEED_PERCENTAGE,
        'actual_percentage': len(seeds_df) / TOTAL_CHANNELS,
    }
    
    # Add quantile stats if available
    if 'quantile' in seeds_df.columns:
        seed_info['quantile_distribution'] = {str(k): int(v) for k, v in seeds_df['quantile'].value_counts().sort_index().items()}
    
    # Add forward_count stats
    if 'forward_count' in seeds_df.columns:
        seed_info['forward_count_stats'] = {
            'min': int(seeds_df['forward_count'].min()),
            'max': int(seeds_df['forward_count'].max()),
            'mean': float(seeds_df['forward_count'].mean()),
            'median': float(seeds_df['forward_count'].median()),
        }
    
    info_file = f"{output_dir}/seed_info.json"
    with open(info_file, 'w') as f:
        json.dump(seed_info, f, indent=2)
    print(f"Saved: {info_file}")
    
    # Save full seed data (for analysis)
    full_file = f"{output_dir}/seeds_full.csv.gz"
    seeds_df.to_csv(full_file, index=False, compression='gzip')
    print(f"Saved: {full_file}")
    
    return seed_info

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresholds', nargs='+', type=int, default=[20, 40, 60, 80])
    parser.add_argument('--skip-forward-count', action='store_true',
                        help='Skip forward count computation (use cached)')
    args = parser.parse_args()
    
    print("="*60)
    print("SEED CREATION FOR TGDATASET EXPERIMENTS")
    print("="*60)
    print(f"Target seeds: ~{TARGET_SEEDS} ({TARGET_SEED_PERCENTAGE*100:.0f}% of {TOTAL_CHANNELS})")
    print(f"Thresholds: {args.thresholds}")
    print(f"Experiments: pure, mixed")
    print(f"GPT Model: {MODEL_NAME}")
    
    # Step 1: Load topic mapping
    print("\n" + "="*60)
    print("STEP 1: Load topic mapping")
    print("="*60)
    df_topics = load_topic_mapping()
    
    # Step 2: Classify gaming topics with GPT
    print("\n" + "="*60)
    print("STEP 2: Classify gaming topics (GPT - strict)")
    print("="*60)
    all_topics = df_topics['topic'].unique().tolist()
    gaming_topics = classify_gaming_topics(all_topics)
    print(f"\nGaming topics identified: {gaming_topics}")
    
    # Step 3: Split channels into gaming vs non-gaming
    print("\n" + "="*60)
    print("STEP 3: Split channels")
    print("="*60)
    gaming_mask = df_topics['topic'].isin(gaming_topics)
    df_gaming = df_topics[gaming_mask].copy()
    df_non_gaming = df_topics[~gaming_mask].copy()
    
    print(f"Gaming channels: {len(df_gaming)}")
    print(f"Non-gaming channels: {len(df_non_gaming)}")
    
    # Rename column for consistency
    df_gaming = df_gaming.rename(columns={'ch_ID': 'channel_id'})
    df_non_gaming = df_non_gaming.rename(columns={'ch_ID': 'channel_id'})
    
    # Step 4: Compute forward counts
    print("\n" + "="*60)
    print("STEP 4: Compute forward counts")
    print("="*60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    forward_cache_file = f"{OUTPUT_DIR}/forward_counts_cache.json"
    
    if args.skip_forward_count and os.path.exists(forward_cache_file):
        print("Loading cached forward counts...")
        with open(forward_cache_file, 'r') as f:
            forward_counts = {int(k): v for k, v in json.load(f).items()}
    else:
        # Load channel mapping
        channel_mapping = load_or_build_channel_mapping()
        
        # Compute forward counts for gaming channels
        gaming_ids = df_gaming['channel_id'].tolist()
        forward_counts = compute_forward_counts(gaming_ids, channel_mapping)
        
        # Cache
        with open(forward_cache_file, 'w') as f:
            json.dump(forward_counts, f)
        print(f"Cached forward counts to {forward_cache_file}")
    
    df_gaming['forward_count'] = df_gaming['channel_id'].map(forward_counts).fillna(0).astype(int)
    
    # Forward count stats
    print(f"\nForward count statistics (gaming channels):")
    print(f"  Min: {df_gaming['forward_count'].min()}")
    print(f"  Max: {df_gaming['forward_count'].max()}")
    print(f"  Mean: {df_gaming['forward_count'].mean():.1f}")
    print(f"  Median: {df_gaming['forward_count'].median()}")
    print(f"  Channels with 0 forwards: {(df_gaming['forward_count'] == 0).sum()}")
    
    # Step 5: Create seeds for each experiment
    print("\n" + "="*60)
    print("STEP 5: Create seed sets")
    print("="*60)
    
    all_seed_info = []
    
    for threshold in args.thresholds:
        for exp_type in ['pure', 'mixed']:
            info = create_seeds(exp_type, threshold, df_gaming, df_non_gaming)
            all_seed_info.append(info)
    
    # Step 6: Save global summaries
    print("\n" + "="*60)
    print("STEP 6: Save global summaries")
    print("="*60)
    
    # Gaming channels details
    gaming_details = {
        'channel_ids': df_gaming['channel_id'].tolist(),
        'total': len(df_gaming),
        'by_topic': {str(k): int(v) for k, v in df_gaming.groupby('topic').size().items()},
        'forward_count_by_quantile': {}
    }
    
    # Create quantiles for summary
    df_gaming_copy = df_gaming.copy()
    df_gaming_copy['quantile'] = pd.qcut(
        df_gaming_copy['forward_count'].rank(method='first'), 
        q=N_QUANTILES, 
        labels=range(N_QUANTILES)
    )
    
    for q in range(N_QUANTILES):
        q_df = df_gaming_copy[df_gaming_copy['quantile'] == q]
        gaming_details['forward_count_by_quantile'][f'Q{q}'] = {
            'count': len(q_df),
            'min_forward': int(q_df['forward_count'].min()) if len(q_df) > 0 else 0,
            'max_forward': int(q_df['forward_count'].max()) if len(q_df) > 0 else 0,
            'mean_forward': float(q_df['forward_count'].mean()) if len(q_df) > 0 else 0,
        }
    
    with open(f"{OUTPUT_DIR}/gaming_channels_details.json", 'w') as f:
        json.dump(gaming_details, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/gaming_channels_details.json")
    
    # Non-gaming channels details
    non_gaming_details = {
        'channel_ids': df_non_gaming['channel_id'].tolist(),
        'total': len(df_non_gaming),
        'by_topic': {str(k): int(v) for k, v in df_non_gaming.groupby('topic').size().items()},
    }
    
    with open(f"{OUTPUT_DIR}/non_gaming_channels_details.json", 'w') as f:
        json.dump(non_gaming_details, f, indent=2)
    print(f"Saved: {OUTPUT_DIR}/non_gaming_channels_details.json")
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n{'Experiment':<25} {'Seeds':>8} {'Gaming':>8} {'Non-Gaming':>10} {'%':>8}")
    print("-"*60)
    for info in all_seed_info:
        print(f"{info['experiment_name']:<25} {info['total_seeds']:>8} {info['gaming_seeds']:>8} {info.get('non_gaming_seeds', 0):>10} {info['actual_percentage']*100:>7.2f}%")
    
    # Save summary
    summary_file = f"{OUTPUT_DIR}/seed_creation_summary.json"
    with open(summary_file, 'w') as f:
        json.dump({
            'target_percentage': TARGET_SEED_PERCENTAGE,
            'target_seeds': TARGET_SEEDS,
            'total_channels': TOTAL_CHANNELS,
            'gaming_topics': gaming_topics,
            'total_gaming_channels': len(df_gaming),
            'total_non_gaming_channels': len(df_non_gaming),
            'gpt_model': MODEL_NAME,
            'experiments': all_seed_info
        }, f, indent=2)
    print(f"\nSaved summary: {summary_file}")

if __name__ == "__main__":
    main()