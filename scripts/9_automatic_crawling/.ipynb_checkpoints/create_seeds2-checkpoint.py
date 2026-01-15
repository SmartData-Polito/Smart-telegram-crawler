#!/usr/bin/env python3
"""
create_seeds2.py
Create seeds for TGDataset experiments using stratified sampling based on gaming_links_count.

gaming_links_count = number of channels labeled "Videogame modding" that a channel points to
                     (via forwarded_from_id)

Creates two types of seed sets:
- PURE_STRATIFIED: 10% of gaming channels, stratified by gaming_links_count
- MIXED_STRATIFIED: same gaming seeds + equal number of non-gaming (other labeled topics)

Thresholds: 10, 15, 20, 40
"""

import os
import sys
import json
import tarfile
import argparse
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm

# ======================== CONFIGURATION ========================
TGDATASET_DIR = "../../material"
LABELED_DATA_DIR = f"{TGDATASET_DIR}/TGDataset/labeled_data"
OUTPUT_DIR = "../../results/experiments_tgdataset"

SAMPLE_FRACTION = 0.10  # 10% per quantile = 10% totale
N_QUANTILES = 10
GAMING_TOPIC = "Videogame modding"

# ======================== HELPER FUNCTIONS ========================

def load_topic_mapping():
    """Load channel to topic mapping."""
    topic_file = f"{LABELED_DATA_DIR}/ch_to_topic_mapping.csv"
    df = pd.read_csv(topic_file)
    return df


def load_or_build_channel_mapping():
    """Load or build mapping from channel_id to tar file and JSON path."""
    mapping_file = f"{TGDATASET_DIR}/channel_file_mapping.json"
    
    if os.path.exists(mapping_file):
        print(f"  Loading existing channel mapping...")
        with open(mapping_file, 'r') as f:
            return json.load(f)
    
    print("  Building channel to file mapping (this takes ~30 min first time)...")
    mapping = {}
    
    for i in range(1, 5):
        tar_path = f"{TGDATASET_DIR}/TGDataset_{i}.tar.gz"
        if not os.path.exists(tar_path):
            print(f"    [SKIP] {tar_path} not found")
            continue
            
        print(f"    Scanning TGDataset_{i}.tar.gz...")
        
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
    
    print(f"    Saved mapping: {len(mapping)} channels")
    return mapping


def compute_gaming_links_count(gaming_channel_ids, channel_mapping):
    """
    For each gaming channel, count how many OTHER gaming channels it points to.
    
    gaming_links_count = number of unique gaming channels in forwarded_from_id
    """
    print(f"\n  Computing gaming_links_count for {len(gaming_channel_ids)} channels...")
    
    gaming_set = set(str(ch) for ch in gaming_channel_ids)
    gaming_links = {}
    
    # Group channels by tar file for efficiency
    by_tar = defaultdict(list)
    for ch_id in gaming_channel_ids:
        ch_id_str = str(ch_id)
        if ch_id_str in channel_mapping:
            info = channel_mapping[ch_id_str]
            by_tar[info['tar']].append((ch_id_str, info['file']))
        else:
            gaming_links[int(ch_id)] = 0
    
    for tar_name, items in by_tar.items():
        tar_path = f"{TGDATASET_DIR}/{tar_name}"
        print(f"    Processing {tar_name}...")
        
        # Group by file
        by_file = defaultdict(list)
        for ch_id, file_path in items:
            by_file[file_path].append(ch_id)
        
        with tarfile.open(tar_path, 'r:gz') as tar:
            for file_path, ch_ids_in_file in tqdm(by_file.items(), desc=f"    {tar_name}"):
                try:
                    f = tar.extractfile(file_path)
                    if f:
                        data = json.load(f)
                        for ch_id in ch_ids_in_file:
                            if ch_id in data:
                                channel_data = data[ch_id]
                                text_messages = channel_data.get('text_messages', {})
                                
                                # Get unique forwarded_from_id that are gaming channels
                                forwarded_gaming = set()
                                for msg in text_messages.values():
                                    fwd_id = msg.get('forwarded_from_id')
                                    if fwd_id is not None:
                                        fwd_id_str = str(fwd_id)
                                        if fwd_id_str in gaming_set and fwd_id_str != ch_id:
                                            forwarded_gaming.add(fwd_id_str)
                                
                                gaming_links[int(ch_id)] = len(forwarded_gaming)
                except Exception as e:
                    pass
    
    # Channels not found get 0
    for ch_id in gaming_channel_ids:
        if ch_id not in gaming_links:
            gaming_links[ch_id] = 0
    
    return gaming_links


def stratified_sample_by_quantile(df, sample_fraction, stratify_col='gaming_links_count'):
    """
    Stratified sampling: take sample_fraction from each quantile.
    
    With 10 quantiles and 10% fraction, we get 10% of total data.
    """
    df = df.copy()
    
    # Create quantile labels
    # Use rank to handle ties
    df['quantile'] = pd.qcut(
        df[stratify_col].rank(method='first'),
        q=N_QUANTILES,
        labels=range(N_QUANTILES)
    )
    
    # Sample from each quantile
    sampled_list = []
    for q in range(N_QUANTILES):
        q_df = df[df['quantile'] == q]
        n_sample = max(1, int(len(q_df) * sample_fraction))  # At least 1
        sampled = q_df.sample(n=min(n_sample, len(q_df)), random_state=42)
        sampled_list.append(sampled)
    
    result = pd.concat(sampled_list, ignore_index=True)
    return result, df  # Return both sampled and full df with quantiles


def create_seeds(experiment_type, threshold, gaming_seeds_df, non_gaming_df):
    """
    Create seed set for an experiment.
    
    experiment_type: 'pure_stratified' or 'mixed_stratified'
    """
    experiment_name = f"threshold_{threshold}_{experiment_type}"
    
    print(f"\n  {'─'*60}")
    print(f"  CREATING: {experiment_name}")
    print(f"  {'─'*60}")
    
    if experiment_type == 'pure_stratified':
        seeds_df = gaming_seeds_df.copy()
        seeds_df['seed_type'] = 'gaming'
        
    else:  # mixed_stratified
        # Gaming seeds (already sampled)
        gaming_part = gaming_seeds_df.copy()
        gaming_part['seed_type'] = 'gaming'
        
        # Non-gaming: same number as gaming, random sample
        n_non_gaming = len(gaming_part)
        if len(non_gaming_df) <= n_non_gaming:
            non_gaming_part = non_gaming_df.copy()
        else:
            non_gaming_part = non_gaming_df.sample(n=n_non_gaming, random_state=42)
        non_gaming_part['seed_type'] = 'non_gaming'
        
        seeds_df = pd.concat([gaming_part, non_gaming_part], ignore_index=True)
    
    # Print info
    n_gaming = len(seeds_df[seeds_df['seed_type'] == 'gaming'])
    n_non_gaming = len(seeds_df[seeds_df['seed_type'] == 'non_gaming']) if 'seed_type' in seeds_df.columns else 0
    
    print(f"    Gaming seeds: {n_gaming}")
    print(f"    Non-gaming seeds: {n_non_gaming}")
    print(f"    Total: {len(seeds_df)}")
    
    # Create output directory
    output_dir = f"{OUTPUT_DIR}/{experiment_name}/level_0"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save nodes file (only channel_id - pipeline doesn't see labels)
    nodes_df = seeds_df[['channel_id']].copy()
    nodes_file = f"{output_dir}/nodes_level_0.csv.gz"
    nodes_df.to_csv(nodes_file, index=False, compression='gzip')
    print(f"    Saved: {nodes_file}")
    
    # Save detailed seed info
    seed_info = {
        'experiment_name': experiment_name,
        'experiment_type': experiment_type,
        'threshold': threshold,
        'total_seeds': len(seeds_df),
        'gaming_seeds': n_gaming,
        'non_gaming_seeds': n_non_gaming,
        'sample_fraction': SAMPLE_FRACTION,
        'n_quantiles': N_QUANTILES,
        'stratify_column': 'gaming_links_count',
    }
    
    # Add quantile distribution for gaming seeds
    if 'quantile' in seeds_df.columns:
        gaming_only = seeds_df[seeds_df['seed_type'] == 'gaming']
        seed_info['quantile_distribution'] = {
            str(k): int(v) for k, v in gaming_only['quantile'].value_counts().sort_index().items()
        }
    
    # Add gaming_links_count stats
    if 'gaming_links_count' in seeds_df.columns:
        gaming_only = seeds_df[seeds_df['seed_type'] == 'gaming']
        seed_info['gaming_links_count_stats'] = {
            'min': int(gaming_only['gaming_links_count'].min()),
            'max': int(gaming_only['gaming_links_count'].max()),
            'mean': float(gaming_only['gaming_links_count'].mean()),
            'median': float(gaming_only['gaming_links_count'].median()),
        }
    
    info_file = f"{output_dir}/seed_info.json"
    with open(info_file, 'w') as f:
        json.dump(seed_info, f, indent=2)
    print(f"    Saved: {info_file}")
    
    # Save full seed data (for analysis)
    full_file = f"{output_dir}/seeds_full.csv.gz"
    seeds_df.to_csv(full_file, index=False, compression='gzip')
    print(f"    Saved: {full_file}")
    
    return seed_info


# ======================== MAIN ========================

def main():
    parser = argparse.ArgumentParser(description='Create stratified seeds based on gaming_links_count')
    parser.add_argument('--thresholds', nargs='+', type=int, default=[10, 15, 20, 40],
                        help='Thresholds for experiments (default: 10 15 20 40)')
    parser.add_argument('--sample-fraction', type=float, default=0.10,
                        help='Fraction to sample from each quantile (default: 0.10)')
    parser.add_argument('--n-quantiles', type=int, default=10,
                        help='Number of quantiles (default: 10)')
    parser.add_argument('--skip-compute', action='store_true',
                        help='Skip gaming_links_count computation (use cached)')
    args = parser.parse_args()
    
    global SAMPLE_FRACTION, N_QUANTILES
    SAMPLE_FRACTION = args.sample_fraction
    N_QUANTILES = args.n_quantiles
    
    print("="*70)
    print(" SEED CREATION v2 - STRATIFIED BY GAMING_LINKS_COUNT")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Sample fraction: {SAMPLE_FRACTION*100:.0f}% per quantile")
    print(f"  Number of quantiles: {N_QUANTILES}")
    print(f"  Thresholds: {args.thresholds}")
    print(f"  Experiment types: pure_stratified, mixed_stratified")
    
    # ========== STEP 1: Load topic mapping ==========
    print("\n" + "="*70)
    print(" STEP 1: Load topic mapping")
    print("="*70)
    
    df_topics = load_topic_mapping()
    print(f"\n  Total labeled channels: {len(df_topics)}")
    print(f"\n  Topics distribution:")
    for topic, count in df_topics['topic'].value_counts().items():
        marker = " ← GAMING" if topic == GAMING_TOPIC else ""
        print(f"    {topic}: {count}{marker}")
    
    # ========== STEP 2: Split gaming vs non-gaming ==========
    print("\n" + "="*70)
    print(" STEP 2: Split gaming vs non-gaming")
    print("="*70)
    
    gaming_mask = df_topics['topic'] == GAMING_TOPIC
    df_gaming = df_topics[gaming_mask].copy()
    df_non_gaming = df_topics[~gaming_mask].copy()
    
    # Rename column
    df_gaming = df_gaming.rename(columns={'ch_ID': 'channel_id'})
    df_non_gaming = df_non_gaming.rename(columns={'ch_ID': 'channel_id'})
    
    print(f"\n  Gaming channels ('{GAMING_TOPIC}'): {len(df_gaming)}")
    print(f"  Non-gaming channels (other topics): {len(df_non_gaming)}")
    
    # ========== STEP 3: Compute gaming_links_count ==========
    print("\n" + "="*70)
    print(" STEP 3: Compute gaming_links_count")
    print("="*70)
    print(f"\n  gaming_links_count = number of unique '{GAMING_TOPIC}' channels")
    print(f"                       that a channel points to via forwarded_from_id")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    cache_file = f"{OUTPUT_DIR}/gaming_links_count_cache.json"
    
    if args.skip_compute and os.path.exists(cache_file):
        print(f"\n  Loading cached gaming_links_count...")
        with open(cache_file, 'r') as f:
            gaming_links = {int(k): v for k, v in json.load(f).items()}
    else:
        channel_mapping = load_or_build_channel_mapping()
        gaming_ids = df_gaming['channel_id'].tolist()
        gaming_links = compute_gaming_links_count(gaming_ids, channel_mapping)
        
        # Cache
        with open(cache_file, 'w') as f:
            json.dump(gaming_links, f)
        print(f"\n  Cached to: {cache_file}")
    
    df_gaming['gaming_links_count'] = df_gaming['channel_id'].map(gaming_links).fillna(0).astype(int)
    
    # ========== STEP 4: Statistics ==========
    print("\n" + "="*70)
    print(" STEP 4: gaming_links_count statistics")
    print("="*70)
    
    glc = df_gaming['gaming_links_count']
    print(f"\n  Total gaming channels: {len(df_gaming)}")
    print(f"\n  gaming_links_count distribution:")
    print(f"    Min:    {glc.min()}")
    print(f"    Max:    {glc.max()}")
    print(f"    Mean:   {glc.mean():.2f}")
    print(f"    Median: {glc.median():.1f}")
    print(f"    Std:    {glc.std():.2f}")
    
    # Count by value
    print(f"\n  Channels with gaming_links_count = 0: {(glc == 0).sum()} ({100*(glc == 0).sum()/len(glc):.1f}%)")
    print(f"  Channels with gaming_links_count > 0: {(glc > 0).sum()} ({100*(glc > 0).sum()/len(glc):.1f}%)")
    print(f"  Channels with gaming_links_count > 5: {(glc > 5).sum()} ({100*(glc > 5).sum()/len(glc):.1f}%)")
    print(f"  Channels with gaming_links_count > 10: {(glc > 10).sum()} ({100*(glc > 10).sum()/len(glc):.1f}%)")
    
    # ========== STEP 5: Create quantiles ==========
    print("\n" + "="*70)
    print(" STEP 5: Quantile distribution")
    print("="*70)
    
    df_gaming_copy = df_gaming.copy()
    df_gaming_copy['quantile'] = pd.qcut(
        df_gaming_copy['gaming_links_count'].rank(method='first'),
        q=N_QUANTILES,
        labels=range(N_QUANTILES)
    )
    
    print(f"\n  {'Quantile':<10} {'Range':<15} {'Channels':<12} {'%':<8} {'Sample (10%)':<12}")
    print(f"  {'-'*57}")
    
    total_sample = 0
    for q in range(N_QUANTILES):
        q_df = df_gaming_copy[df_gaming_copy['quantile'] == q]
        min_val = q_df['gaming_links_count'].min()
        max_val = q_df['gaming_links_count'].max()
        count = len(q_df)
        pct = 100 * count / len(df_gaming_copy)
        sample_n = max(1, int(count * SAMPLE_FRACTION))
        total_sample += sample_n
        
        if min_val == max_val:
            range_str = f"{min_val}"
        else:
            range_str = f"{min_val}-{max_val}"
        
        print(f"  Q{q:<9} {range_str:<15} {count:<12} {pct:<8.1f} {sample_n:<12}")
    
    print(f"  {'-'*57}")
    print(f"  {'TOTAL':<10} {'':<15} {len(df_gaming_copy):<12} {'100.0':<8} {total_sample:<12}")
    
    # ========== STEP 6: Stratified sampling ==========
    print("\n" + "="*70)
    print(" STEP 6: Stratified sampling")
    print("="*70)
    
    gaming_seeds, df_gaming_with_q = stratified_sample_by_quantile(
        df_gaming, 
        SAMPLE_FRACTION, 
        'gaming_links_count'
    )
    
    print(f"\n  Sampled {len(gaming_seeds)} gaming channels ({100*len(gaming_seeds)/len(df_gaming):.1f}% of {len(df_gaming)})")
    
    # Verify sampling
    print(f"\n  Verification - sampled per quantile:")
    print(f"  {'Quantile':<10} {'Original':<12} {'Sampled':<12} {'%':<8}")
    print(f"  {'-'*42}")
    
    for q in range(N_QUANTILES):
        orig_count = len(df_gaming_with_q[df_gaming_with_q['quantile'] == q])
        samp_count = len(gaming_seeds[gaming_seeds['quantile'] == q])
        pct = 100 * samp_count / orig_count if orig_count > 0 else 0
        print(f"  Q{q:<9} {orig_count:<12} {samp_count:<12} {pct:<8.1f}")
    
    # Sampled stats
    print(f"\n  Sampled gaming_links_count stats:")
    print(f"    Min:    {gaming_seeds['gaming_links_count'].min()}")
    print(f"    Max:    {gaming_seeds['gaming_links_count'].max()}")
    print(f"    Mean:   {gaming_seeds['gaming_links_count'].mean():.2f}")
    print(f"    Median: {gaming_seeds['gaming_links_count'].median():.1f}")
    
    # ========== STEP 7: Create experiments ==========
    print("\n" + "="*70)
    print(" STEP 7: Create experiments")
    print("="*70)
    
    all_seed_info = []
    
    for threshold in args.thresholds:
        for exp_type in ['pure_stratified', 'mixed_stratified']:
            info = create_seeds(exp_type, threshold, gaming_seeds, df_non_gaming)
            all_seed_info.append(info)
    
    # ========== STEP 8: Summary ==========
    print("\n" + "="*70)
    print(" SUMMARY")
    print("="*70)
    
    print(f"\n  {'Experiment':<35} {'Gaming':<10} {'Non-Gaming':<12} {'Total':<10}")
    print(f"  {'-'*67}")
    for info in all_seed_info:
        print(f"  {info['experiment_name']:<35} {info['gaming_seeds']:<10} {info['non_gaming_seeds']:<12} {info['total_seeds']:<10}")
    
    # Save global summary
    summary = {
        'sample_fraction': SAMPLE_FRACTION,
        'n_quantiles': N_QUANTILES,
        'thresholds': args.thresholds,
        'gaming_topic': GAMING_TOPIC,
        'total_gaming_channels': len(df_gaming),
        'total_non_gaming_channels': len(df_non_gaming),
        'sampled_gaming_channels': len(gaming_seeds),
        'gaming_links_count_stats': {
            'population': {
                'min': int(df_gaming['gaming_links_count'].min()),
                'max': int(df_gaming['gaming_links_count'].max()),
                'mean': float(df_gaming['gaming_links_count'].mean()),
                'median': float(df_gaming['gaming_links_count'].median()),
            },
            'sample': {
                'min': int(gaming_seeds['gaming_links_count'].min()),
                'max': int(gaming_seeds['gaming_links_count'].max()),
                'mean': float(gaming_seeds['gaming_links_count'].mean()),
                'median': float(gaming_seeds['gaming_links_count'].median()),
            }
        },
        'experiments': all_seed_info
    }
    
    summary_file = f"{OUTPUT_DIR}/seed_creation_v2_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved summary: {summary_file}")
    
    print("\n" + "="*70)
    print(" DONE")
    print("="*70)


if __name__ == "__main__":
    main()