#!/usr/bin/env python3
"""
MASTER ORCHESTRATOR: Run the complete political topic detection pipeline.

Usage:
    python master_orchestrator.py --start-level 0 --max-levels 10
    python master_orchestrator.py --level 0
    python master_orchestrator.py --level non_visited
    python master_orchestrator.py --level 0 --threshold 0.5
    python master_orchestrator.py --level 0 --start-date 2024-01-01 --end-date 2024-06-30
    python master_orchestrator.py --start-level 0 --max-levels 10 --start-date 2024-01-01 --end-date 2024-06-30 --threshold 0.4
    usare --force per rifare tutto forzatamente
"""

import os
import sys
import time
import argparse
import subprocess
import json

import pandas as pd

# ======================== CONFIGURATION ========================
DEFAULT_POLITICS_THRESHOLD = 0.40
MAX_LEVELS = 10

PIPELINE_STEPS = [
    ("step1_preprocess.py", "Preprocessing"),
    ("step2_lda_train.py", "LDA Training"),
    ("step3_extract_topics.py", "Topic Extraction"),
    ("step4_classify_topics.py", "Topic Classification (ChatGPT)"),
    ("step5_compute_politics_ratio.py", "Politics Ratio Computation"),
]

# ======================== TIMING ========================
GLOBAL_START = time.perf_counter()
LEVEL_TIMES = {}

def log_time(message: str, prefix: str = "") -> None:
    elapsed = time.perf_counter() - GLOBAL_START
    print(f"[{elapsed:8.2f}s]{prefix} {message}")

def log_header(message: str) -> None:
    print("\n" + "=" * 70)
    log_time(message)
    print("=" * 70)

# ======================== STEP EXECUTION ========================
def run_step(script_name: str, level: str, extra_args: list = None) -> tuple:
    """Run a pipeline step and return (success, elapsed_time)."""
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    
    if not os.path.exists(script_path):
        log_time(f"ERROR: Script not found: {script_path}", prefix=" [ERROR]")
        return False, 0
    
    cmd = [sys.executable, script_path, "--level", level]
    if extra_args:
        cmd.extend(extra_args)
    
    log_time(f"Running: {' '.join(cmd)}", prefix=" [RUN]")
    
    step_start = time.perf_counter()
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        elapsed = time.perf_counter() - step_start
        return result.returncode == 0, elapsed
    except subprocess.CalledProcessError as e:
        elapsed = time.perf_counter() - step_start
        log_time(f"Step failed with return code {e.returncode}", prefix=" [ERROR]")
        return False, elapsed
    except Exception as e:
        elapsed = time.perf_counter() - step_start
        log_time(f"Step failed: {e}", prefix=" [ERROR]")
        return False, elapsed

def check_step_completed(level: str, step_num: int) -> bool:
    """Check if a step has already been completed."""
    base_dir = f"../../results/levels_automatic/level_{level}"
    
    completion_files = {
        1: f"{base_dir}/preprocessing/step1_completed.txt",
        2: f"{base_dir}/lda/step2_completed.txt",
        3: f"{base_dir}/topics/step3_completed.txt",
        4: f"{base_dir}/classification/step4_completed.txt",
        5: f"{base_dir}/channel_analysis/step5_completed.txt",
    }
    
    return os.path.exists(completion_files.get(step_num, ""))

def get_next_level_node_count(level: str) -> int:
    """Get the number of nodes prepared for the next level."""
    try:
        next_level = int(level) + 1
        nodes_path = f"../../results/levels_automatic/level_{next_level}/nodes_level_{next_level}.csv.gz"
        
        if not os.path.exists(nodes_path):
            return -1
        
        df = pd.read_csv(nodes_path, compression='gzip')
        return len(df)
    except ValueError:
        return -1

def check_english_messages_exist(level: str) -> int:
    """Check how many English messages exist for this level."""
    english_file = f"../../results/levels_automatic/level_{level}/preprocessing/messages_english_clean.tsv.gz"
    
    if not os.path.exists(english_file):
        return -1
    
    try:
        df = pd.read_csv(english_file, sep='\t', compression='gzip')
        return len(df)
    except Exception as e:
        log_time(f"Error reading English file: {e}", prefix=" [WARN]")
        return -1

# ======================== LEVEL PROCESSING ========================
def process_level(level: str, threshold: float, start_date: str = None, 
                  end_date: str = None, skip_completed: bool = True) -> bool:
    """Process a single level through all pipeline steps."""
    log_header(f"PROCESSING LEVEL {level}")
    log_time(f"  Threshold: {threshold*100:.0f}%")
    if start_date:
        log_time(f"  Start date: {start_date}")
    if end_date:
        log_time(f"  End date: {end_date}")
    
    level_start = time.perf_counter()
    step_times = {}
    
    # Run each step
    for step_num, (script, description) in enumerate(PIPELINE_STEPS, 1):
        log_time(f"\n--- Step {step_num}/{len(PIPELINE_STEPS)}: {description} ---", prefix=" [STEP]")
        
        # Check if already completed
        if skip_completed and check_step_completed(level, step_num):
            log_time(f"Already completed, skipping", prefix=" [SKIP]")
            step_times[f"step{step_num}_{description.lower().replace(' ', '_')}"] = 0
            continue
        
        # Build extra args for each step
        extra_args = []
        
        # Step 1: preprocessing - pass date filters
        if step_num == 1:
            if start_date:
                extra_args.extend(["--start-date", start_date])
            if end_date:
                extra_args.extend(["--end-date", end_date])
            
            success, elapsed = run_step(script, level, extra_args)
            step_times[f"step{step_num}_preprocessing"] = elapsed
            
            if not success:
                log_time(f"Step {step_num} failed!", prefix=" [FAIL]")
                return False
            
            english_count = check_english_messages_exist(level)
            if english_count == 0:
                log_time(f"No English messages at level {level}, skipping LDA steps", prefix=" [WARN]")
                log_time(f"Jumping directly to next level preparation", prefix=" [INFO]")
                break
            elif english_count > 0:
                log_time(f"Found {english_count} English messages", prefix=" [INFO]")
            
            continue
        
        # Step 5: politics ratio - pass threshold
        if script == "step5_compute_politics_ratio.py":
            extra_args = ["--threshold", str(threshold)]
        
        success, elapsed = run_step(script, level, extra_args)
        step_times[f"step{step_num}_{description.lower().replace(' ', '_').replace('(', '').replace(')', '')}"] = elapsed
        
        if not success:
            log_time(f"Step {step_num} failed!", prefix=" [FAIL]")
            return False
    
    # Prepare next level (solo per livelli numerici)
    try:
        level_int = int(level)
        log_time(f"\n--- Step 6/6: Preparing Next Level ---", prefix=" [STEP]")
        
        political_channels_file = f"../../results/levels_automatic/level_{level}/channel_analysis/political_channels.json"
        
        if os.path.exists(political_channels_file):
            # Pass threshold to step6
            extra_args = ["--threshold", str(threshold)]
            success, elapsed = run_step("step6_prepare_next_level.py", level, extra_args)
            step_times["step6_prepare_next_level"] = elapsed
        else:
            log_time("No political channels file found (no English messages?)", prefix=" [WARN]")
            log_time("Creating empty next level", prefix=" [INFO]")
            
            next_level = level_int + 1
            next_level_dir = f"../../results/levels_automatic/level_{next_level}"
            os.makedirs(next_level_dir, exist_ok=True)
            
            df_empty = pd.DataFrame(columns=['type_and_id'])
            df_empty.to_csv(f"{next_level_dir}/nodes_level_{next_level}.csv.gz", 
                           index=False, compression='gzip')
            
            step_times["step6_prepare_next_level"] = 0
            success = True
    except ValueError:
        log_time(f"\n--- Skipping Step 6 (non-numeric level) ---", prefix=" [INFO]")
        success = True
    
    level_time = time.perf_counter() - level_start
    step_times["total"] = level_time
    
    LEVEL_TIMES[level] = step_times
    
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
    parser.add_argument("--level", type=str, default=None,
                        help="Run single level only (can be number or name like 'non_visited')")
    parser.add_argument("--threshold", type=float, default=DEFAULT_POLITICS_THRESHOLD,
                        help=f"Politics threshold (default: {DEFAULT_POLITICS_THRESHOLD})")
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date filter (YYYY-MM-DD)")
    parser.add_argument("--skip-completed", action="store_true", default=True,
                        help="Skip already completed steps (default: True)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-run all steps")
    args = parser.parse_args()
    
    log_header("POLITICAL TOPIC DETECTION PIPELINE")
    log_time(f"Configuration:")
    log_time(f"  Politics threshold: {args.threshold*100:.0f}%")
    log_time(f"  Max levels: {args.max_levels}")
    if args.start_date:
        log_time(f"  Start date: {args.start_date}")
    if args.end_date:
        log_time(f"  End date: {args.end_date}")
    
    os.makedirs("../../results/levels_automatic", exist_ok=True)
    
    # Save config
    config = {
        "threshold": args.threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "max_levels": args.max_levels
    }
    config_path = "../../results/levels_automatic/pipeline_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # Single level mode
    if args.level is not None:
        level = str(args.level)
        success = process_level(level, args.threshold, args.start_date, 
                               args.end_date, skip_completed=not args.force)
        
        total_time = time.perf_counter() - GLOBAL_START
        
        level_summary = {
            "level": level,
            "status": "COMPLETED" if success else "FAILED",
            "threshold": args.threshold,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "total_time_seconds": total_time,
            "step_times": LEVEL_TIMES.get(level, {})
        }
        
        summary_path = f"../../results/levels_automatic/level_{level}/level_summary.json"
        with open(summary_path, "w") as f:
            json.dump(level_summary, f, indent=2)
        log_time(f"Level summary saved to {summary_path}")
        
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
        
        nodes_path = f"../../results/levels_automatic/level_{level_str}/nodes_level_{level_str}.csv.gz"
        
        if current_level > 0 and not os.path.exists(nodes_path):
            log_time(f"No nodes file for level {current_level}, stopping iteration")
            break
        
        if current_level > 0:
            df_nodes = pd.read_csv(nodes_path, compression='gzip')
            if len(df_nodes) == 0:
                log_time(f"Level {current_level} has 0 nodes, stopping iteration")
                break
        
        success = process_level(level_str, args.threshold, args.start_date,
                               args.end_date, skip_completed=not args.force)
        
        if not success:
            log_header(f"PIPELINE STOPPED AT LEVEL {current_level}")
            sys.exit(1)
        
        levels_processed += 1
        
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
    
    log_time("\nTiming breakdown by level:")
    for level, times in LEVEL_TIMES.items():
        log_time(f"  Level {level}:")
        for step_name, step_time in times.items():
            if step_time > 0:
                log_time(f"    {step_name}: {step_time:.2f}s")
    
    summary = {
        "levels_processed": levels_processed,
        "start_level": args.start_level,
        "final_level": current_level - 1 if current_level > args.start_level else args.start_level,
        "threshold": args.threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
        "total_time_seconds": total_time,
        "level_times": LEVEL_TIMES
    }
    
    summary_path = "../../results/levels_automatic/pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    log_time(f"Summary saved to {summary_path}")

if __name__ == "__main__":
    main()