#!/usr/bin/env python3
"""
Master orchestrator for TGDataset gaming detection pipeline.
WITH TIMING TRACKING: salva tempi per esperimento, livello e step.
"""

import os
import sys
import time
import argparse
import subprocess
import json
from datetime import datetime

import pandas as pd

DEFAULT_THRESHOLD = 0.40
MAX_LEVELS = 10

# Step 6 ora è incluso nella lista
PIPELINE_STEPS = [
    ("step1_preprocess.py", "Preprocessing"),
    ("step2_lda_train.py", "LDA Training"),
    ("step3_extract_topics.py", "Topic Extraction"),
    ("step4_classify_topics.py", "Topic Classification (Gaming)"),
    ("step5_compute_gaming_ratio.py", "Gaming Ratio Computation"),
    ("step6_prepare_next_level.py", "Prepare Next Level"),
]

GLOBAL_START = time.perf_counter()

# ============================================================
# NUOVA STRUTTURA: Dizionario globale per tracking tempi
# ============================================================
# Esempio struttura finale:
# TIMING_DATA = {
#     "experiment_name": "threshold_20_pure",
#     "threshold": 0.20,
#     "start_time": "2025-02-03 14:30:00",
#     "levels": {
#         "0": {
#             "nodes": 245,
#             "steps": {"step1_preprocess.py": 180.5, ...},
#             "total_time_seconds": 4500.2
#         }
#     }
# }
TIMING_DATA = {
    "levels": {}
}


def log_time(msg):
    """Stampa messaggio con timestamp dall'inizio."""
    elapsed = time.perf_counter() - GLOBAL_START
    print(f"[{elapsed:8.2f}s] {msg}")


def format_time(seconds):
    """
    Converte secondi in formato leggibile.
    Esempi: 45.2 -> "45s", 125.5 -> "2m 5s", 3725.0 -> "1h 2m 5s"
    """
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def run_step(script, level, base_dir, extra_args=None):
    """
    Esegue uno step della pipeline.
    
    MODIFICA: ora restituisce (success, elapsed_time) invece di solo success.
    """
    cmd = [sys.executable, script, "--level", str(level), "--base-dir", base_dir]
    if extra_args:
        cmd.extend(extra_args)
    
    log_time(f"Running: {' '.join(cmd)}")
    
    # Misura tempo di esecuzione
    step_start = time.perf_counter()
    result = subprocess.run(cmd)
    step_elapsed = time.perf_counter() - step_start
    
    success = result.returncode == 0
    
    # Log tempo impiegato
    log_time(f"  -> {'OK' if success else 'FAILED'} in {format_time(step_elapsed)}")
    
    return success, step_elapsed


def process_level(level, threshold, base_dir, start_date=None, end_date=None, num_nodes=0):
    """
    Processa un singolo livello.
    
    MODIFICA: traccia tempi di ogni step in TIMING_DATA.
    """
    
    log_time(f"\n{'='*60}")
    log_time(f"PROCESSING LEVEL {level}")
    log_time(f"{'='*60}")
    
    # Inizializza struttura per questo livello
    level_key = str(level)
    level_start_time = datetime.now()
    
    TIMING_DATA["levels"][level_key] = {
        "start_time": level_start_time.strftime("%Y-%m-%d %H:%M:%S"),
        "nodes": num_nodes,
        "steps": {},
        "status": "RUNNING"
    }
    
    level_start = time.perf_counter()
    
    for step_num, (script, description) in enumerate(PIPELINE_STEPS, 1):
        log_time(f"\n--- Step {step_num}: {description} ---")
        
        extra_args = []
        
        # Step 1: Preprocessing - aggiunge date filter se presenti
        if step_num == 1:
            if start_date:
                extra_args.extend(["--start-date", start_date])
            if end_date:
                extra_args.extend(["--end-date", end_date])
        
        # Step 5: Gaming ratio - aggiunge threshold
        if step_num == 5:
            extra_args.extend(["--threshold", str(threshold)])
        
        # Step 6: Prepare next level - aggiunge threshold
        if step_num == 6:
            extra_args.extend(["--threshold", str(threshold)])
        
        # Esegui step e ottieni tempo
        success, step_time = run_step(script, level, base_dir, extra_args)
        
        # Salva tempo dello step
        TIMING_DATA["levels"][level_key]["steps"][script] = step_time
        
        if not success:
            log_time(f"[FAIL] Step {step_num} failed!")
            TIMING_DATA["levels"][level_key]["status"] = "FAILED"
            TIMING_DATA["levels"][level_key]["failed_step"] = script
            return False
    
    # Calcola tempo totale del livello
    level_elapsed = time.perf_counter() - level_start
    level_end_time = datetime.now()
    
    TIMING_DATA["levels"][level_key]["end_time"] = level_end_time.strftime("%Y-%m-%d %H:%M:%S")
    TIMING_DATA["levels"][level_key]["total_time_seconds"] = level_elapsed
    TIMING_DATA["levels"][level_key]["total_time_human"] = format_time(level_elapsed)
    TIMING_DATA["levels"][level_key]["status"] = "COMPLETED"
    
    log_time(f"\nLevel {level} completed in {format_time(level_elapsed)}")
    
    return True


def save_timing_report(base_dir):
    """
    Salva il report dei tempi in JSON.
    File: {base_dir}/timing_report.json
    """
    output_file = f"{base_dir}/timing_report.json"
    
    with open(output_file, 'w') as f:
        json.dump(TIMING_DATA, f, indent=2)
    
    log_time(f"Saved timing report: {output_file}")
    return output_file


def print_timing_summary():
    """
    Stampa tabella riassuntiva dei tempi a fine esecuzione.
    """
    print(f"\n{'='*90}")
    print("TIMING SUMMARY")
    print(f"{'='*90}")
    
    # Header
    print(f"\n{'Level':<7} {'Nodes':>7}  {'Step1':>9} {'Step2':>11} {'Step3':>7} {'Step4':>8} {'Step5':>7} {'Step6':>7}  {'TOTAL':>12}")
    print("─" * 90)
    
    total_experiment_time = 0
    
    for level_key in sorted(TIMING_DATA["levels"].keys(), key=int):
        level_data = TIMING_DATA["levels"][level_key]
        nodes = level_data.get("nodes", 0)
        steps = level_data.get("steps", {})
        level_total = level_data.get("total_time_seconds", 0)
        total_experiment_time += level_total
        
        # Estrai tempi per ogni step (usa 0 se non presente)
        t1 = steps.get("step1_preprocess.py", 0)
        t2 = steps.get("step2_lda_train.py", 0)
        t3 = steps.get("step3_extract_topics.py", 0)
        t4 = steps.get("step4_classify_topics.py", 0)
        t5 = steps.get("step5_compute_gaming_ratio.py", 0)
        t6 = steps.get("step6_prepare_next_level.py", 0)
        
        # Formatta e stampa riga
        print(f"{level_key:<7} {nodes:>7}  {format_time(t1):>9} {format_time(t2):>11} {format_time(t3):>7} {format_time(t4):>8} {format_time(t5):>7} {format_time(t6):>7}  {format_time(level_total):>12}")
    
    print("─" * 90)
    print(f"{'TOTAL':<7} {'':<7}  {'':<9} {'':<11} {'':<7} {'':<8} {'':<7} {'':<7}  {format_time(total_experiment_time):>12}")
    print()


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
    
    # ============================================================
    # Inizializza TIMING_DATA con metadati esperimento
    # ============================================================
    experiment_start_time = datetime.now()
    
    TIMING_DATA["experiment_name"] = args.experiment_name
    TIMING_DATA["threshold"] = args.threshold
    TIMING_DATA["start_time"] = experiment_start_time.strftime("%Y-%m-%d %H:%M:%S")
    TIMING_DATA["start_date_filter"] = args.start_date
    TIMING_DATA["end_date_filter"] = args.end_date
    TIMING_DATA["status"] = "RUNNING"
    
    log_time(f"{'='*60}")
    log_time(f"TGDATASET GAMING PIPELINE - {args.experiment_name}")
    log_time(f"Threshold: {args.threshold*100:.0f}%")
    log_time(f"{'='*60}")
    
    # Save config (come prima)
    os.makedirs(base_dir, exist_ok=True)
    config = {
        "experiment_name": args.experiment_name,
        "threshold": args.threshold,
        "start_date": args.start_date,
        "end_date": args.end_date,
    }
    with open(f"{base_dir}/pipeline_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    current_level = args.start_level
    levels_completed = 0
    
    while current_level < args.start_level + args.max_levels:
        nodes_file = f"{base_dir}/level_{current_level}/nodes_level_{current_level}.csv.gz"
        
        if not os.path.exists(nodes_file):
            log_time(f"No nodes file for level {current_level}, stopping")
            break
        
        df = pd.read_csv(nodes_file, compression='gzip')
        num_nodes = len(df)
        
        if num_nodes == 0:
            log_time(f"Level {current_level} has 0 nodes, stopping")
            break
        
        log_time(f"Level {current_level}: {num_nodes} nodes")
        
        # Passa num_nodes a process_level per il tracking
        success = process_level(current_level, args.threshold, base_dir,
                               args.start_date, args.end_date, num_nodes)
        
        if not success:
            log_time(f"Pipeline failed at level {current_level}")
            TIMING_DATA["status"] = "FAILED"
            TIMING_DATA["failed_at_level"] = current_level
            
            # Salva report anche in caso di fallimento
            experiment_end_time = datetime.now()
            total_time = time.perf_counter() - GLOBAL_START
            TIMING_DATA["end_time"] = experiment_end_time.strftime("%Y-%m-%d %H:%M:%S")
            TIMING_DATA["total_time_seconds"] = total_time
            TIMING_DATA["total_time_human"] = format_time(total_time)
            TIMING_DATA["levels_completed"] = levels_completed
            
            save_timing_report(base_dir)
            print_timing_summary()
            sys.exit(1)
        
        levels_completed += 1
        current_level += 1
    
    # ============================================================
    # Fine esperimento: salva tempi e stampa summary
    # ============================================================
    experiment_end_time = datetime.now()
    total_time = time.perf_counter() - GLOBAL_START
    
    TIMING_DATA["end_time"] = experiment_end_time.strftime("%Y-%m-%d %H:%M:%S")
    TIMING_DATA["total_time_seconds"] = total_time
    TIMING_DATA["total_time_human"] = format_time(total_time)
    TIMING_DATA["levels_completed"] = levels_completed
    TIMING_DATA["status"] = "COMPLETED"
    
    log_time(f"\n{'='*60}")
    log_time(f"PIPELINE COMPLETED")
    log_time(f"Total time: {format_time(total_time)}")
    log_time(f"{'='*60}")
    
    # Salva report JSON
    save_timing_report(base_dir)
    
    # Stampa tabella riassuntiva
    print_timing_summary()


if __name__ == "__main__":
    main()