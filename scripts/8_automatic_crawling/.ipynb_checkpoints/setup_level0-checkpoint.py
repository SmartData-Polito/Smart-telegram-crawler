#!/usr/bin/env python3
"""
SETUP: Initialize level 0 with seed channels.

This script creates the initial nodes file for level 0 from your seed channels.

Usage:
    python setup_level0.py --seeds-file path/to/seeds.csv
    python setup_level0.py --seeds "channel_1234,channel_5678"
"""

import os
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(description="Setup level 0 with seed channels")
    parser.add_argument("--seeds-file", type=str, default=None,
                        help="CSV file with seed channels (column: type_and_id)")
    parser.add_argument("--seeds", type=str, default=None,
                        help="Comma-separated list of seed channel IDs")
    args = parser.parse_args()
    
    # Create output directory
    output_dir = "../results/levels_automatic/level_0"
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = f"{output_dir}/nodes_level_0.csv.gz"
    
    # Load or parse seeds
    if args.seeds_file:
        if not os.path.exists(args.seeds_file):
            print(f"ERROR: Seeds file not found: {args.seeds_file}")
            return
        
        df = pd.read_csv(args.seeds_file)
        if 'type_and_id' not in df.columns:
            # Try to find the right column
            possible_cols = ['channel_id', 'id', 'node_id', 'type_and_id']
            for col in possible_cols:
                if col in df.columns:
                    df = df.rename(columns={col: 'type_and_id'})
                    break
        
        seeds = df['type_and_id'].tolist()
        
    elif args.seeds:
        seeds = [s.strip() for s in args.seeds.split(',') if s.strip()]
        
    else:
        print("ERROR: Provide either --seeds-file or --seeds")
        print("\nExample usage:")
        print("  python setup_level0.py --seeds-file ../data/seed_channels.csv")
        print("  python setup_level0.py --seeds 'channel_123,channel_456'")
        return
    
    # Create nodes dataframe
    df_nodes = pd.DataFrame({'type_and_id': seeds})
    df_nodes.to_csv(output_path, index=False, compression="gzip")
    
    print(f"Created level 0 with {len(seeds)} seed channels")
    print(f"Output: {output_path}")
    print(f"\nFirst 10 channels:")
    for ch in seeds[:10]:
        print(f"  - {ch}")
    if len(seeds) > 10:
        print(f"  ... and {len(seeds) - 10} more")
    
    print(f"\nReady to run: python master_orchestrator.py --start-level 0")

if __name__ == "__main__":
    main()