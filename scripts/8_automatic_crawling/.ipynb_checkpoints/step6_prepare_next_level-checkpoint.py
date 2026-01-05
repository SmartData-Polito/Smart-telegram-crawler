#!/usr/bin/env python3
"""
STEP 6: Prepare nodes for next level (children of political channels).
Usage: python step6_prepare_next_level.py --level 0
       python step6_prepare_next_level.py --level 0 --base-dir ../../results/experiments/peak_jul_aug
       python step6_prepare_next_level.py --level 0 --threshold 0.5
"""

import os
import time
import argparse
import json
import pandas as pd

# ======================== TIMING ========================
START_TIME = time.perf_counter()
STEP_TIMES = {}

def log_time(msg: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.2f}s] {msg}")

def start_timer(name: str) -> float:
    return time.perf_counter()

def end_timer(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    return elapsed

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, required=True)
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override threshold (if different from step5)")
    args = parser.parse_args()
    
    current_level = int(args.level)
    next_level = current_level + 1
    base_dir = args.base_dir
    
    log_time(f"Preparing nodes for level {next_level} from level {current_level}")
    log_time(f"  Base dir: {base_dir}")
    
    # Paths
    current_level_dir = f"{base_dir}/level_{current_level}"
    next_level_dir = f"{base_dir}/level_{next_level}"
    
    political_channels_path = f"{current_level_dir}/channel_analysis/political_channels.json"
    channel_stats_path = f"{current_level_dir}/channel_analysis/channel_politics_stats.csv"
    edges_path = "../../material/discovery_edges.csv.gz"
    
    os.makedirs(next_level_dir, exist_ok=True)
    next_nodes_path = f"{next_level_dir}/nodes_level_{next_level}.csv.gz"
    excluded_nodes_path = f"{next_level_dir}/excluded_nodes.json"
    
    # Load political channels data
    t_start = start_timer("load_political_channels")
    if not os.path.exists(political_channels_path):
        log_time(f"ERROR: Political channels not found: {political_channels_path}")
        return
    
    with open(political_channels_path, "r") as f:
        politics_data = json.load(f)
    
    # Get threshold - use passed argument or from file
    original_threshold = politics_data.get("threshold", 0.4)
    threshold = args.threshold if args.threshold is not None else original_threshold
    
    log_time(f"Using threshold: {threshold*100:.0f}%")
    
    # If threshold is different, recompute political channels from stats
    if args.threshold is not None and args.threshold != original_threshold:
        log_time(f"Threshold changed from {original_threshold*100:.0f}% to {threshold*100:.0f}%")
        log_time("Recomputing political channels from channel stats...")
        
        if not os.path.exists(channel_stats_path):
            log_time(f"ERROR: Channel stats not found: {channel_stats_path}")
            return
        
        df_stats = pd.read_csv(channel_stats_path)
        political_channels = set(
            df_stats[df_stats['political_ratio'] >= threshold]['channel_id'].tolist()
        )
        log_time(f"Political channels at {threshold*100:.0f}%: {len(political_channels)}")
    else:
        # Use precomputed political channels
        if "details" in politics_data and "political_channels" in politics_data["details"]:
            political_channels = set(politics_data["details"]["political_channels"])
        else:
            political_channels = set(politics_data.get("political_channels", []))
    
    log_time(f"Loaded {len(political_channels)} political channels from level {current_level}")
    end_timer("load_political_channels", t_start)
    
    # Load discovery edges
    t_start = start_timer("load_edges")
    if not os.path.exists(edges_path):
        log_time(f"ERROR: Discovery edges file not found: {edges_path}")
        log_time("Creating empty next level nodes file")
        pd.DataFrame(columns=['type_and_id']).to_csv(next_nodes_path, index=False, compression="gzip")
        return
    
    log_time("Loading discovery edges...")
    df_edges = pd.read_csv(edges_path, compression='gzip')
    df_edges = df_edges.dropna(subset=["parent", "type_and_id"])
    df_edges = df_edges.drop_duplicates(subset=["parent", "type_and_id"])
    log_time(f"Loaded {len(df_edges)} edges")
    end_timer("load_edges", t_start)
    
    # Collect all processed nodes
    t_start = start_timer("collect_processed_nodes")
    all_processed_nodes = set()
    for lvl in range(current_level + 1):
        lvl_nodes_path = f"{base_dir}/level_{lvl}/nodes_level_{lvl}.csv.gz"
        if os.path.exists(lvl_nodes_path):
            df_lvl = pd.read_csv(lvl_nodes_path, compression="gzip")
            if 'type_and_id' in df_lvl.columns:
                all_processed_nodes.update(df_lvl['type_and_id'].tolist())
    
    log_time(f"Total nodes processed in levels 0-{current_level}: {len(all_processed_nodes)}")
    end_timer("collect_processed_nodes", t_start)
    
    # Find children of political channels
    t_start = start_timer("find_children")
    children = df_edges[
        df_edges["parent"].isin(political_channels)
    ]["type_and_id"].dropna().unique().tolist()
    
    children_set = set(children)
    log_time(f"Found {len(children_set)} children of political channels")
    end_timer("find_children", t_start)
    
    # Filter out already processed
    t_start = start_timer("filter_processed")
    new_children = children_set - all_processed_nodes
    excluded_loops = children_set & all_processed_nodes
    
    log_time(f"New children (not in previous levels): {len(new_children)}")
    log_time(f"Excluded (would create loops): {len(excluded_loops)}")
    end_timer("filter_processed", t_start)
    
    # Save next level nodes
    t_start = start_timer("save_nodes")
    df_next = pd.DataFrame({'type_and_id': sorted(new_children)})
    df_next.to_csv(next_nodes_path, index=False, compression="gzip")
    log_time(f"Saved {len(df_next)} nodes for level {next_level} to {next_nodes_path}")
    end_timer("save_nodes", t_start)
    
    # Save excluded info
    t_start = start_timer("save_excluded")
    excluded_info = {
        "level": next_level,
        "parent_level": current_level,
        "threshold_used": threshold,
        "political_channels_count": len(political_channels),
        "total_children": len(children_set),
        "new_children": len(new_children),
        "excluded_loops": len(excluded_loops),
        "excluded_node_ids": sorted(excluded_loops)
    }
    
    with open(excluded_nodes_path, "w") as f:
        json.dump(excluded_info, f, indent=2)
    log_time(f"Saved excluded nodes info to {excluded_nodes_path}")
    end_timer("save_excluded", t_start)
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"\nCOMPLETED in {total_time:.2f}s")
    
    if len(new_children) == 0:
        log_time("\n*** No new nodes for next level - pipeline complete! ***")
    else:
        log_time(f"\n*** Ready for level {next_level} with {len(new_children)} nodes ***")
    
    with open(f"{next_level_dir}/step6_completed.txt", "w") as f:
        f.write(f"Step 6: Prepare Next Level\n")
        f.write(f"Level: {current_level} -> {next_level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Threshold used: {threshold}\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Political channels from level {current_level}: {len(political_channels)}\n")
        f.write(f"  Total children found: {len(children_set)}\n")
        f.write(f"  New nodes for level {next_level}: {len(new_children)}\n")
        f.write(f"  Excluded (loops): {len(excluded_loops)}\n")

if __name__ == "__main__":
    main()