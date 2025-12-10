#!/usr/bin/env python3
"""
Crea un livello con tutti i canali NON visitati dalla pipeline automatica.
Poi puoi eseguire la pipeline normale su questo livello.

Usage: python create_non_visited_level.py

Output: results/levels_automatic/level_non_visited/nodes_level_non_visited.csv.gz
"""

import os
import json
import sqlite3
import pandas as pd
from glob import glob

def main():
    print("=" * 60)
    print("CREATING NON-VISITED LEVEL")
    print("=" * 60)
    
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
    print("\n2. Loading visited channels from pipeline...")
    
    base_dir = "../../results/levels_automatic"
    level_dirs = sorted(glob(f"{base_dir}/level_[0-9]*"))
    
    visited_channels = set()
    
    for level_dir in level_dirs:
        level_name = os.path.basename(level_dir)
        tracking_file = f"{level_dir}/preprocessing/channels_tracking.json"
        
        if not os.path.exists(tracking_file):
            # Prova a leggere dal nodes file
            nodes_file = f"{level_dir}/nodes_{level_name}.csv.gz"
            if os.path.exists(nodes_file):
                df_nodes = pd.read_csv(nodes_file, compression='gzip')
                visited_channels.update(df_nodes['type_and_id'].tolist())
                print(f"   {level_name}: {len(df_nodes)} nodes (from nodes file)")
            continue
        
        with open(tracking_file, 'r') as f:
            tracking = json.load(f)
        
        # Tutti i canali che sono stati INPUT del livello
        total_nodes = tracking.get("total_nodes", 0)
        
        # Leggi dal nodes file per avere la lista completa
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
        "total_channels_in_db": len(all_channels),
        "visited_by_pipeline": len(visited_channels),
        "non_visited": len(non_visited),
        "levels_checked": [os.path.basename(d) for d in level_dirs]
    }
    
    with open(f"{output_dir}/level_info.json", 'w') as f:
        json.dump(info, f, indent=2)
    print(f"   Saved: {output_dir}/level_info.json")
    
    # ==================== 5. INSTRUCTIONS ====================
    print("\n" + "=" * 60)
    print("DONE! Now run the pipeline on non_visited level:")
    print("=" * 60)
    print(f"""
    python master_orchestrator.py --level non_visited
    
    This will:
    1. Preprocess {len(non_visited)} channels
    2. Train LDA model
    3. Extract topics  
    4. Classify topics (politics/non-politics)
    5. Compute political ratio per channel
    
    Results will be in: {output_dir}/
    """)

if __name__ == "__main__":
    main()