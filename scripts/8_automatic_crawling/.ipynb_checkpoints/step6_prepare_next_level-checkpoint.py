#!/usr/bin/env python3
"""
STEP 6: Prepare nodes for next level (children of political channels).
Usage: python step6_prepare_next_level.py --level 0

Reads from: channel_analysis/political_channels.json
Output: level_{N+1}/nodes_level_{N+1}.csv.gz
"""

import os
import time
import argparse
import json
import pandas as pd

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Prepare next level nodes")
    parser.add_argument("--level", type=str, required=True, help="Current hierarchy level")
    args = parser.parse_args()
    
    current_level = int(args.level)
    next_level = current_level + 1
    log_time(f"Preparing nodes for level {next_level} from level {current_level}")
    
    # Paths
    base_dir = f"../../results/levels_automatic"
    current_level_dir = f"{base_dir}/level_{current_level}"
    next_level_dir = f"{base_dir}/level_{next_level}"
    
    # Input files - now reads from channel_analysis/
    political_channels_path = f"{current_level_dir}/channel_analysis/political_channels.json"
    hierarchy_path = "../../../../telegram_2024/usc-tg-24-us-election/hierarchy.csv"
    
    # Output files
    os.makedirs(next_level_dir, exist_ok=True)
    next_nodes_path = f"{next_level_dir}/nodes_level_{next_level}.csv.gz"
    excluded_nodes_path = f"{next_level_dir}/excluded_nodes.json"
    
    # Load political channels from current level
    if not os.path.exists(political_channels_path):
        log_time(f"ERROR: Political channels not found: {political_channels_path}")
        return
    
    with open(political_channels_path, "r") as f:
        politics_data = json.load(f)
    
    political_channels = set(politics_data["political_channels"])
    log_time(f"Loaded {len(political_channels)} political channels from level {current_level}")
    
    # Load hierarchy
    if not os.path.exists(hierarchy_path):
        log_time(f"ERROR: Hierarchy file not found: {hierarchy_path}")
        log_time("Creating empty next level nodes file")
        pd.DataFrame(columns=['type_and_id']).to_csv(next_nodes_path, index=False, compression="gzip")
        return
    
    log_time("Loading hierarchy...")
    df_hierarchy = pd.read_csv(hierarchy_path)
    log_time(f"Loaded hierarchy with {len(df_hierarchy)} edges")
    
    # Collect all nodes processed in previous levels
    all_processed_nodes = set()
    for lvl in range(current_level + 1):
        lvl_nodes_path = f"{base_dir}/level_{lvl}/nodes_level_{lvl}.csv.gz"
        if os.path.exists(lvl_nodes_path):
            df_lvl = pd.read_csv(lvl_nodes_path, compression="gzip")
            if 'type_and_id' in df_lvl.columns:
                all_processed_nodes.update(df_lvl['type_and_id'].tolist())
    
    log_time(f"Total nodes processed in levels 0-{current_level}: {len(all_processed_nodes)}")
    
    # Find parent/child columns
    parent_col = None
    child_col = None
    
    for p, c in [('parent_id', 'child_id'), ('source', 'target'), 
                  ('from', 'to'), ('parent', 'child')]:
        if p in df_hierarchy.columns and c in df_hierarchy.columns:
            parent_col, child_col = p, c
            break
    
    if parent_col is None:
        log_time(f"ERROR: Could not identify parent/child columns in hierarchy")
        log_time(f"Available columns: {df_hierarchy.columns.tolist()}")
        return
    
    log_time(f"Using columns: parent={parent_col}, child={child_col}")
    
    # Find children
    children = set()
    for parent in political_channels:
        mask = df_hierarchy[parent_col] == parent
        channel_children = df_hierarchy.loc[mask, child_col].tolist()
        children.update(channel_children)
    
    log_time(f"Found {len(children)} children of political channels")
    
    # Filter out already processed
    new_children = children - all_processed_nodes
    excluded_loops = children & all_processed_nodes
    
    log_time(f"New children (not in previous levels): {len(new_children)}")
    log_time(f"Excluded (would create loops): {len(excluded_loops)}")
    
    # Save
    df_next = pd.DataFrame({'type_and_id': sorted(new_children)})
    df_next.to_csv(next_nodes_path, index=False, compression="gzip")
    log_time(f"Saved {len(df_next)} nodes for level {next_level} to {next_nodes_path}")
    
    excluded_info = {
        "level": next_level,
        "parent_level": current_level,
        "total_children": len(children),
        "new_children": len(new_children),
        "excluded_loops": len(excluded_loops),
        "excluded_node_ids": sorted(excluded_loops)
    }
    
    with open(excluded_nodes_path, "w") as f:
        json.dump(excluded_info, f, indent=2)
    
    total_time = time.perf_counter() - START_TIME
    log_time(f"\nCOMPLETED in {total_time:.2f}s")
    
    if len(new_children) == 0:
        log_time("\n*** No new nodes for next level - pipeline complete! ***")
    else:
        log_time(f"\n*** Ready for level {next_level} with {len(new_children)} nodes ***")
    
    with open(f"{next_level_dir}/step6_completed.txt", "w") as f:
        f.write(f"Next level preparation completed in {total_time:.2f}s\n")
        f.write(f"New nodes: {len(new_children)}\n")
        f.write(f"Excluded (loops): {len(excluded_loops)}\n")

if __name__ == "__main__":
    main()
