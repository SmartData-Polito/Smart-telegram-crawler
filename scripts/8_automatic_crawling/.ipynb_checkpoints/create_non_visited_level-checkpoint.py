#!/usr/bin/env python3
"""
Crea un livello con tutti i canali NON visitati per ogni esperimento threshold.

Usage: python create_non_visited_experiments.py --experiment-name threshold_20
       python create_non_visited_experiments.py --all
"""

import os
import json
import sqlite3
import argparse
import pandas as pd
from glob import glob

def create_non_visited_for_experiment(exp_name):
    print("=" * 60)
    print(f" CREATING NON-VISITED LEVEL FOR: {exp_name}")
    print("=" * 60)
    
    base_dir = f"../../results/experiments/{exp_name}"
    
    if not os.path.exists(base_dir):
        print(f"  [ERROR] Experiment not found: {base_dir}")
        return False
    
    # ==================== 1. LOAD ALL CHANNELS FROM CHATS.DB ====================
    print("\n1. Loading all channels from chats.db...")
    
    chats_path = '../../material/chats.db'
    conn = sqlite3.connect(chats_path)
    df_chats = pd.read_sql_query("SELECT DISTINCT type_and_id FROM chats", conn)
    conn.close()
    
    df_chats = df_chats.dropna(subset=['type_and_id'])
    all_channels = set(df_chats['type_and_id'].tolist())
    print(f"   Total channels in chats.db: {len(all_channels)}")
    
    # ==================== 2. LOAD ALL VISITED CHANNELS ====================
    print(f"\n2. Loading visited channels from {exp_name}...")
    
    level_dirs = sorted(glob(f"{base_dir}/level_[0-9]*"))
    
    visited_channels = set()
    
    for level_dir in level_dirs:
        level_name = os.path.basename(level_dir)
        nodes_file = f"{level_dir}/nodes_{level_name}.csv.gz"
        
        if os.path.exists(nodes_file):
            df_nodes = pd.read_csv(nodes_file, compression='gzip')
            visited_channels.update(df_nodes['type_and_id'].tolist())
            print(f"   {level_name}: {len(df_nodes)} nodes")
    
    print(f"\n   Total visited channels: {len(visited_channels)}")
    
    # ==================== 3. COMPUTE NON-VISITED ====================
    print("\n3. Computing non-visited channels...")
    
    non_visited = all_channels - visited_channels
    print(f"   All channels:     {len(all_channels)}")
    print(f"   Visited:          {len(visited_channels)}")
    print(f"   Non-visited:      {len(non_visited)}")
    
    # ==================== 4. CREATE LEVEL DIRECTORY ====================
    print("\n4. Creating non_visited level...")
    
    output_dir = f"{base_dir}/level_non_visited"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save nodes file
    df_non_visited = pd.DataFrame({'type_and_id': sorted(non_visited)})
    nodes_path = f"{output_dir}/nodes_level_non_visited.csv.gz"
    df_non_visited.to_csv(nodes_path, index=False, compression='gzip')
    print(f"   Saved: {nodes_path}")
    print(f"   Channels to process: {len(df_non_visited)}")
    
    # Save info file
    info = {
        "experiment": exp_name,
        "total_channels_in_db": len(all_channels),
        "visited_by_pipeline": len(visited_channels),
        "non_visited": len(non_visited),
        "levels_checked": [os.path.basename(d) for d in level_dirs]
    }
    
    with open(f"{output_dir}/level_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"   Saved: {output_dir}/level_info.json")
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment-name', type=str, help='Nome esperimento (es: threshold_20)')
    parser.add_argument('--all', action='store_true', help='Processa tutti i threshold')
    args = parser.parse_args()
    
    if args.all:
        experiments = ["threshold_20", "threshold_40", "threshold_60", "threshold_80"]
        for exp in experiments:
            create_non_visited_for_experiment(exp)
            print("\n")
    elif args.experiment_name:
        create_non_visited_for_experiment(args.experiment_name)
    else:
        print("Usage:")
        print("  python create_non_visited_experiments.py --experiment-name threshold_20")
        print("  python create_non_visited_experiments.py --all")

if __name__ == "__main__":
    main()
