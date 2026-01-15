#!/usr/bin/env python3
"""
Extract and display all results from TGDataset experiments.
Run from: ~/Thesis/Smart_crawler_telegram/scripts/9_automatic_crawling/
"""

import json
import pandas as pd
import os
import json

BASE_DIR = "../../results/experiments_tgdataset"

def main():
    print("="*70)
    print(" TGDATASET EXPERIMENTS - RESULTS EXTRACTION")
    print("="*70)
    
    # ============================================================
    # 1. SEED CREATION SUMMARY
    # ============================================================
    print("\n" + "="*70)
    print(" SEED CREATION SUMMARY")
    print("="*70)
    
    summary_file = f"{BASE_DIR}/seed_creation_summary.json"
    if os.path.exists(summary_file):
        with open(summary_file) as f:
            summary = json.load(f)
        
        print(f"\nGaming topics: {summary['gaming_topics']}")
        print(f"Total gaming channels: {summary['total_gaming_channels']}")
        print(f"Total non-gaming channels: {summary['total_non_gaming_channels']}")
        print(f"Target seeds: {summary['target_seeds']} ({summary['target_percentage']*100:.0f}%)")
        print(f"GPT model: {summary['gpt_model']}")
        
        print(f"\n{'Experiment':<25} {'Seeds':>8} {'Gaming':>8} {'Non-Gaming':>10} {'%':>8}")
        print("-"*65)
        for exp in summary['experiments']:
            pct = exp['actual_percentage'] * 100
            print(f"{exp['experiment_name']:<25} {exp['total_seeds']:>8} {exp['gaming_seeds']:>8} {exp['non_gaming_seeds']:>10} {pct:>7.2f}%")
    else:
        print(f"[WARN] {summary_file} not found")
    
    # ============================================================
    # 2. GAMING CHANNELS DETAILS
    # ============================================================
    print("\n" + "="*70)
    print(" GAMING CHANNELS DETAILS")
    print("="*70)
    
    gaming_file = f"{BASE_DIR}/gaming_channels_details.json"
    if os.path.exists(gaming_file):
        with open(gaming_file) as f:
            gaming = json.load(f)
        
        print(f"\nTotal gaming channels: {gaming['total']}")
        print(f"By topic: {gaming['by_topic']}")
        
        print(f"\n{'Quantile':<10} {'Count':>8} {'Min Fwd':>10} {'Max Fwd':>10} {'Mean Fwd':>10}")
        print("-"*55)
        for q, stats in gaming['forward_count_by_quantile'].items():
            print(f"{q:<10} {stats['count']:>8} {stats['min_forward']:>10} {stats['max_forward']:>10} {stats['mean_forward']:>10.1f}")
    else:
        print(f"[WARN] {gaming_file} not found")
    
    # ============================================================
    # 3. EXPERIMENT RESULTS
    # ============================================================
    print("\n" + "="*70)
    print(" EXPERIMENT RESULTS")
    print("="*70)
    
    experiments = [d for d in os.listdir(BASE_DIR) 
                   if os.path.isdir(f"{BASE_DIR}/{d}") and d.startswith("threshold_")]
    
    if not experiments:
        print("\n[INFO] No experiments found yet. Run the pipeline first.")
        return
    
    results = []
    for exp in sorted(experiments):
        analysis_file = f"{BASE_DIR}/{exp}/experiment_analysis.json"
        if os.path.exists(analysis_file):
            with open(analysis_file) as f:
                analysis = json.load(f)
            
            g = analysis['global']
            gt = g.get('ground_truth', {})
            
            results.append({
                'experiment': exp,
                'levels': len(analysis['levels']),
                'nodes': g.get('total_nodes', 0),
                'found': g.get('channels_found', 0),
                'gaming': g.get('gaming_channels', 0),
                'precision': gt.get('precision', 0),
                'recall': gt.get('recall', 0),
                'f1': gt.get('f1', 0),
            })
    
    if results:
        print(f"\n{'Experiment':<25} {'Lvl':>4} {'Nodes':>8} {'Found':>8} {'Gaming':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
        print("-"*90)
        for r in results:
            print(f"{r['experiment']:<25} {r['levels']:>4} {r['nodes']:>8} {r['found']:>8} {r['gaming']:>8} {r['precision']*100:>7.1f}% {r['recall']*100:>7.1f}% {r['f1']*100:>7.1f}%")
        
        # Save to CSV
        df = pd.DataFrame(results)
        csv_file = f"{BASE_DIR}/results_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")
    else:
        print("\n[INFO] No completed experiments found. Run analyze_all_levels.py first.")

if __name__ == "__main__":
    main()