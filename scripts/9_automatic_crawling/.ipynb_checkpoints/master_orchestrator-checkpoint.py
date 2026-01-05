#!/usr/bin/env python3
"""
Master orchestrator for TGDataset gaming detection pipeline.
"""

import os
import sys
import time
import argparse
import subprocess
import json

import pandas as pd

DEFAULT_THRESHOLD = 0.40
MAX_LEVELS = 10

PIPELINE_STEPS = [
    ("step1_preprocess.py", "Preprocessing"),
    ("step2_lda_train.py", "LDA Training"),
    ("step3_extract_topics.py", "Topic Extraction"),
    ("step4_classify_topics.py", "Topic Classification (Gaming)"),
    ("step5_compute_gaming_ratio.py", "Gaming Ratio Computation"),
]

GLOBAL_START = time.perf_counter()

def log_time(msg):
    elapsed = time.perf_counter() - GLOBAL_START
    print(f"[{elapsed:8.2f}s] {msg}")

def run_step(script, level, base_dir, extra_args=None):
    """Run a pipeline step."""
    cmd = [sys.executable, script, "--level", str(level), "--base-dir", base_dir]
    if extra_args:
        cmd.extend(extra_args)
    
    log_time(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    return result.returncode == 0

def process_level(level, threshold, base_dir, start_date=None, end_date=None):
    """Process a single level."""
    
    log_time(f"\n{'='*60}")
    log_time(f"PROCESSING LEVEL {level}")
    log_time(f"{'='*60}")
    
    for step_num, (script, description) in enumerate(PIPELINE_STEPS, 1):
        log_time(f"\n--- Step {step_num}: {description} ---")
        
        extra_args = []
        
        if step_num == 1:  # Preprocessing
            if start_date:
                extra_args.extend(["--start-date", start_date])
            if end_date:
                extra_args.extend(["--end-date", end_date])
        
        if step_num == 5:  # Gaming ratio
            extra_args.extend(["--threshold", str(threshold)])
        
        success = run_step(script, level, base_dir, extra_args)
        
        if not success:
            log_time(f"[FAIL] Step {step_num} failed!")
            return False
    
    # Prepare next level
    log_time(f"\n--- Step 6: Prepare Next Level ---")
    success = run_step("step6_prepare_next_level.py", level, base_dir, 
                       ["--threshold", str(threshold)])
    
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-level", type=int, default=0)
    parser.add_argument("--max-levels", type=int, default=MAX_LEVELS)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--experiment-name", type=str, required=True)
    parser.add_argument("--start-date", type=str, default=None)
    parser.add_argument("--end-date", type=str, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    
    base_dir = f"../../results/experiments_tgdataset/{args.experiment_name}"
    
    log_time(f"{'='*60}")
    log_time(f"TGDATASET GAMING PIPELINE - {args.experiment_name}")
    log_time(f"Threshold: {args.threshold*100:.0f}%")
    log_time(f"{'='*60}")
    
    # Save config
    config = {
        "experiment_name": args.experiment_name,
        "threshold": args.threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    with open(f"{base_dir}/pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    current_level = args.start_level
    
    while current_level < args.start_level + args.max_levels:
        nodes_file = f"{base_dir}/level_{current_level}/nodes_level_{current_level}.csv.gz"
        
        if not os.path.exists(nodes_file):
            log_time(f"No nodes file for level {current_level}, stopping")
            break
        
        df = pd.read_csv(nodes_file, compression='gzip')
        if len(df) == 0:
            log_time(f"Level {current_level} has 0 nodes, stopping")
            break
        
        log_time(f"Level {current_level}: {len(df)} nodes")
        
        success = process_level(current_level, args.threshold, base_dir,
                               args.start_date, args.end_date)
        
        if not success:
            log_time(f"Pipeline failed at level {current_level}")
            sys.exit(1)
        
        current_level += 1
    
    total_time = time.perf_counter() - GLOBAL_START
    log_time(f"\n{'='*60}")
    log_time(f"PIPELINE COMPLETED")
    log_time(f"Total time: {total_time/60:.1f} minutes")
    log_time(f"{'='*60}")

if __name__ == "__main__":
    main()