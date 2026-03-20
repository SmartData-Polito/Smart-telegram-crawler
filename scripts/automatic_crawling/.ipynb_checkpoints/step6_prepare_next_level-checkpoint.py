#!/usr/bin/env python3
"""
Step 6: Prepare next level using forwarded_from_id.
For each gaming channel, extract forwarded_from_id from messages.
These become the next level candidates.
"""

import os
import json
import argparse
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

def prepare_next_level(level, base_dir, threshold):
    """Prepare nodes for next level based on forwarded_from_id."""
    
    print(f"\n{'='*60}")
    print(f"STEP 6: PREPARE NEXT LEVEL (from level {level})")
    print(f"{'='*60}")
    
    level_dir = f"{base_dir}/level_{level}"
    next_level = int(level) + 1
    next_level_dir = f"{base_dir}/level_{next_level}"
    
    # Load gaming channels
    gaming_file = f"{level_dir}/channel_analysis/gaming_channels.json"
    if not os.path.exists(gaming_file):
        gaming_file = f"{level_dir}/channel_analysis/political_channels.json"
    
    if not os.path.exists(gaming_file):
        print(f"[ERROR] Gaming channels file not found")
        return
    
    with open(gaming_file, 'r') as f:
        gaming_data = json.load(f)
    
    gaming_channel_ids = set(gaming_data.get('gaming_channel_ids', gaming_data.get('political_channel_ids', [])))
    print(f"Gaming channels to expand: {len(gaming_channel_ids)}")
    
    if not gaming_channel_ids:
        print("No gaming channels, creating empty next level")
        os.makedirs(next_level_dir, exist_ok=True)
        df = pd.DataFrame(columns=['channel_id'])
        df.to_csv(f"{next_level_dir}/nodes_level_{next_level}.csv.gz", 
                  index=False, compression='gzip')
        return
    
    # Load messages from preprocessing
    messages_file = f"{level_dir}/preprocessing/messages_english_clean.tsv.gz"
    df_messages = pd.read_csv(messages_file, sep='\t', compression='gzip')
    
    print(f"Total messages: {len(df_messages)}")
    
    # Filter to gaming channels only
    df_gaming = df_messages[df_messages['channel_id'].isin(gaming_channel_ids)]
    print(f"Messages from gaming channels: {len(df_gaming)}")
    
    # Extract forwarded_from_id
    print("Extracting forwarded_from_id...")
    
    # Get unique forwarded_from_id values (excluding None/NaN)
    forwarded_ids = df_gaming['forwarded_from_id'].dropna().unique()
    
    # Convert to integers (they might be floats due to NaN)
    forwarded_ids = set(int(fid) for fid in forwarded_ids if pd.notna(fid))
    
    print(f"  Unique forwarded_from_id: {len(forwarded_ids)}")
    
    # Remove already visited channels
    visited = set()
    for l in range(next_level):
        nodes_file = f"{base_dir}/level_{l}/nodes_level_{l}.csv.gz"
        if os.path.exists(nodes_file):
            df = pd.read_csv(nodes_file, compression='gzip')
            visited.update(df['channel_id'].astype(int).tolist())
    
    new_channels = forwarded_ids - visited
    print(f"  Already visited: {len(forwarded_ids) - len(new_channels)}")
    print(f"  New channels: {len(new_channels)}")
    
    # Save next level
    os.makedirs(next_level_dir, exist_ok=True)
    
    df_next = pd.DataFrame({'channel_id': list(new_channels)})
    output_file = f"{next_level_dir}/nodes_level_{next_level}.csv.gz"
    df_next.to_csv(output_file, index=False, compression='gzip')
    
    print(f"Saved: {output_file}")
    
    # Save expansion info
    info = {
        'parent_level': level,
        'gaming_channels_expanded': len(gaming_channel_ids),
        'messages_from_gaming': len(df_gaming),
        'unique_forwarded_from': len(forwarded_ids),
        'already_visited': len(forwarded_ids) - len(new_channels),
        'new_channels': len(new_channels)
    }
    
    with open(f"{next_level_dir}/expansion_info.json", 'w') as f:
        json.dump(info, f, indent=2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, required=True)
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--threshold', type=float, default=0.4)
    args = parser.parse_args()
    
    prepare_next_level(args.level, args.base_dir, args.threshold)

if __name__ == "__main__":
    main()