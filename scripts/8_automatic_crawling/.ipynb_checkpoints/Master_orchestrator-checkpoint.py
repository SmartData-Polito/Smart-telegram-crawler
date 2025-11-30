#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR: Run the complete political topic detection pipeline.

This script orchestrates the iterative process:
1. Start with seed channels (level 0)
2. Preprocess messages
3. Train LDA model
4. Extract topics
5. Classify topics (politics/non-politics) via ChatGPT API
6. Compute political ratio per channel
7. Find children of political channels (>=40% political messages)
8. Repeat for next level until no new nodes

Usage:
    python master_orchestrator.py --start-level 0 --max-levels 5
    python master_orchestrator.py --level 0  # Run single level only
"""

import os
import sys
import time
import argparse
import subprocess
import json

# ======================== CONFIGURATION ========================
POLITICS_THRESHOLD = 0.40  # 40% minimum political messages
MAX_LEVELS = 10  # Safety limit

# Scripts to run in order
PIPELINE_STEPS = [
    ("step1_preprocess.py", "Preprocessing"),
    ("step2_lda_train.py", "LDA Training"),
    ("step3_extract_topics.py", "Topic Extraction"),
    ("step4_classify_topics.py", "Topic Classification (ChatGPT)"),
    ("step5_compute_politics_ratio.py", "Politics Ratio Computation"),
]

# ======================== TIMING ========================
GLOBAL_START = time.perf_counter()

def log_time(message: str, prefix: str = "") -> None:
    elapsed = time.perf_counter() - GLOBAL_START
    print(f"[{elapsed:8.2f}s]{prefix} {message}")

def log_header(message: str) -> None:
    print("\n" + "=" * 70)
    log_time(message)
    print("=" * 70)

# ======================== STEP EXECUTION ========================
def run_step(script_name: str, level: str, extra_args: list = None) -> bool:
    """Run a pipeline step and return success status."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        log_time(f"ERROR: Script not found: {script_path}", prefix=" [ERROR]")
        return False
    
    cmd = [sys.executable, script_path, "--level", level]
    if extra_args:
        cmd.extend(extra_args)
    
    log_time(f"Running: {' '.join(cmd)}", prefix=" [RUN]")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        log_time(f"Step failed with return code {e.returncode}", prefix=" [ERROR]")
        return False
    except Exception as e:
        log_time(f"Step failed: {e}", prefix=" [ERROR]")
        return False

def check_step_completed(level: str, step_num: int) -> bool:
    """Check if a step has already been completed."""
    base_dir = f"../results/levels_automatic/level_{level}"
    
    completion_files = {
        1: f"{base_dir}/preprocessing/step1_completed.txt",
        2: f"{base_dir}/lda/step2_completed.txt",
        3: f"{base_dir}/lda/step3_completed.txt",
        4: f"{base_dir}/lda/step4_completed.txt",
        5: f"{base_dir}/lda/step5_completed.txt",
    }
    
    return os.path.exists(completion_files.get(step_num, ""))

def get_next_level_node_count(level: str) -> int:
    """Get the number of nodes prepared for the next level."""
    next_level = int(level) + 1
    nodes_path = f"../results/levels_automatic/level_{next_level}/nodes_level_{next_level}.csv"
    
    if not os.path.exists(nodes_path):
        return -1
    
    import pandas as pd
    df = pd.read_csv(nodes_path)
    return len(df)

# ======================== LEVEL PROCESSING ========================
def process_level(level: str, skip_completed: bool = True) -> bool:
    """Process a single level through all pipeline steps."""
    log_header(f"PROCESSING LEVEL {level}")
    
    level_start = time.perf_counter()
    
    # Run each step
    for step_num, (script, description) in enumerate(PIPELINE_STEPS, 1):
        log_time(f"\n--- Step {step_num}/5: {description} ---", prefix=" [STEP]")
        
        # Check if already completed
        if skip_completed and check_step_completed(level, step_num):
            log_time(f"Already completed, skipping", prefix=" [SKIP]")
            continue
        
        # Run the step
        extra_args = []
        if script == "step5_compute_politics_ratio.py":
            extra_args = ["--threshold", str(POLITICS_THRESHOLD)]
        
        success = run_step(script, level, extra_args)
        
        if not success:
            log_time(f"Step {step_num} failed!", prefix=" [FAIL]")
            return False
    
    # Prepare next level
    log_time(f"\n--- Step 6/6: Preparing Next Level ---", prefix=" [STEP]")
    success = run_step("step6_prepare_next_level.py", level)
    
    level_time = time.perf_counter() - level_start
    log_time(f"\nLevel {level} completed in {level_time:.2f}s", prefix=" [DONE]")
    
    return success

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(
        description="Master orchestrator for political topic detection pipeline"
    )
    parser.add_argument("--start-level", type=int, default=0,
                        help="Starting level (default: 0)")
    parser.add_argument("--max-levels", type=int, default=MAX_LEVELS,
                        help=f"Maximum levels to process (default: {MAX_LEVELS})")
    parser.add_argument("--level", type=int, default=None,
                        help="Run single level only")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already completed steps (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps")
    args = parser.parse_args()
    
    log_header("POLITICAL TOPIC DETECTION PIPELINE")
    log_time(f"Configuration:")
    log_time(f"  Politics threshold: {POLITICS_THRESHOLD*100:.0f}%")
    log_time(f"  Max levels: {args.max_levels}")
    
    # Create base results directory
    os.makedirs("../results/levels_automatic", exist_ok=True)
    
    # Single level mode
    if args.level is not None:
        level = str(args.level)
        success = process_level(level, skip_completed=not args.force)
        
        if success:
            log_header(f"LEVEL {level} COMPLETED SUCCESSFULLY")
        else:
            log_header(f"LEVEL {level} FAILED")
            sys.exit(1)
        return
    
    # Multi-level iteration mode
    current_level = args.start_level
    levels_processed = 0
    
    while current_level < args.start_level + args.max_levels:
        level_str = str(current_level)
        
        # Check if this level has nodes to process
        nodes_path = f"../results/levels_automatic/level_{level_str}/nodes_level_{level_str}.csv"
        
        if current_level > 0 and not os.path.exists(nodes_path):
            log_time(f"No nodes file for level {current_level}, stopping iteration")
            break
        
        if current_level > 0:
            import pandas as pd
            df_nodes = pd.read_csv(nodes_path)
            if len(df_nodes) == 0:
                log_time(f"Level {current_level} has 0 nodes, stopping iteration")
                break
        
        # Process this level
        success = process_level(level_str, skip_completed=not args.force)
        
        if not success:
            log_header(f"PIPELINE STOPPED AT LEVEL {current_level}")
            sys.exit(1)
        
        levels_processed += 1
        
        # Check if there are nodes for next level
        next_node_count = get_next_level_node_count(level_str)
        
        if next_node_count == 0:
            log_time(f"\nNo new nodes for level {current_level + 1}, pipeline complete!")
            break
        elif next_node_count > 0:
            log_time(f"\nNext level {current_level + 1} has {next_node_count} nodes")
        
        current_level += 1
    
    # Final summary
    total_time = time.perf_counter() - GLOBAL_START
    log_header("PIPELINE COMPLETED")
    log_time(f"Total levels processed: {levels_processed}")
    log_time(f"Total time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    
    # Save summary
    summary = {
        "levels_processed": levels_processed,
        "start_level": args.start_level,
        "final_level": current_level - 1 if current_level > args.start_level else args.start_level,
        "politics_threshold": POLITICS_THRESHOLD,
        "total_time_seconds": total_time
    }
    
    summary_path = "../results/levels_automatic/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()