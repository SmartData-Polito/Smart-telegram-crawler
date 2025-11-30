#!/usr/bin/env python3
"""
STEP 6: Prepare nodes for next level (children of political channels).
Usage: python step6_prepare_next_level.py --level 0

This script:
1. Loads political channels from current level
2. Finds their children in the hierarchy
3. Avoids loops (children pointing back to parents)
4. Creates nodes file for next level
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
    base_dir = f"../results/levels_automatic"
    current_level_dir = f"{base_dir}/level_{current_level}"
    next_level_dir = f"{base_dir}/level_{next_level}"
    
    # Input files
    political_channels_path = f"{current_level_dir}/lda/political_channels.json"
    hierarchy_path = "../../../../telegram_2024/usc-tg-24-us-election/hierarchy.csv"  # Adjust path as needed
    
    # Output files
    os.makedirs(next_level_dir, exist_ok=True)
    next_nodes_path = f"{next_level_dir}/nodes_level_{next_level}.csv"
    excluded_nodes_path = f"{next_level_dir}/excluded_nodes.json"
    
    # Load political channels from current level
    if not os.path.exists(political_channels_path):
        log_time(f"ERROR: Political channels not found: {political_channels_path}")
        return
    
    with open(political_channels_path, "r") as f:
        politics_data = json.load(f)
    
    political_channels = set(politics_data["political_channels"])
    log_time(f"Loaded {len(political_channels)} political channels from level {current_level}")
    
    # Load hierarchy to find children
    if not os.path.exists(hierarchy_path):
        log_time(f"ERROR: Hierarchy file not found: {hierarchy_path}")
        log_time("Creating empty next level nodes file")
        pd.DataFrame(columns=['type_and_id']).to_csv(next_nodes_path, index=False)
        return
    
    log_time("Loading hierarchy...")
    df_hierarchy = pd.read_csv(hierarchy_path)
    log_time(f"Loaded hierarchy with {len(df_hierarchy)} edges")
    
    # Collect all nodes processed in previous levels (to avoid loops)
    all_processed_nodes = set()
    for lvl in range(current_level + 1):
        lvl_nodes_path = f"{base_dir}/level_{lvl}/nodes_level_{lvl}.csv"
        if os.path.exists(lvl_nodes_path):
            df_lvl = pd.read_csv(lvl_nodes_path)
            if 'type_and_id' in df_lvl.columns:
                all_processed_nodes.update(df_lvl['type_and_id'].tolist())
    
    log_time(f"Total nodes processed in levels 0-{current_level}: {len(all_processed_nodes)}")
    
    # Find children of political channels
    # Assuming hierarchy has columns: parent_id, child_id (adjust as needed)
    # Common column names: source/target, from/to, parent/child
    
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
    
    # Filter out already processed nodes (avoid loops)
    new_children = children - all_processed_nodes
    excluded_loops = children & all_processed_nodes
    
    log_time(f"New children (not in previous levels): {len(new_children)}")
    log_time(f"Excluded (would create loops): {len(excluded_loops)}")
    
    # Create next level nodes dataframe
    df_next = pd.DataFrame({'type_and_id': sorted(new_children)})
    df_next.to_csv(next_nodes_path, index=False)
    log_time(f"Saved {len(df_next)} nodes for level {next_level} to {next_nodes_path}")
    
    # Save excluded nodes info
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
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    log_time(f"\nCOMPLETED in {total_time:.2f}s")
    
    # Check if we should continue
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