#!/usr/bin/env python3
"""
Extract and display all results from TGDataset experiments.
Run from: ~/Thesis/Smart_crawler_telegram/scripts/9_automatic_crawling/

METRICHE (nuova definizione):
- TP = canali con etichetta "Videogame modding" E classificati gaming
- FP = canali con etichetta ALTRO E classificati gaming  
- FN = canali con etichetta "Videogame modding" E NON classificati (TUTTI, non solo visitati)
- Precision = TP / (TP + FP)
- Recall = TP / (TP + FN)
"""

import json
import pandas as pd
import os

BASE_DIR = "../../results/experiments_tgdataset"

def main():
    print("="*70)
    print(" TGDATASET EXPERIMENTS - RESULTS EXTRACTION")
    print("="*70)
    
    # ============================================================
    # 1. SEED CREATION SUMMARY (v1 - pure/mixed)
    # ============================================================
    print("\n" + "="*70)
    print(" SEED CREATION SUMMARY (v1 - pure/mixed)")
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
        print(f"\n[INFO] {summary_file} not found (v1 seeds not created)")
    
    # ============================================================
    # 2. SEED CREATION SUMMARY (v2 - stratified)
    # ============================================================
    print("\n" + "="*70)
    print(" SEED CREATION SUMMARY (v2 - stratified)")
    print("="*70)
    
    summary_file_v2 = f"{BASE_DIR}/seed_creation_summary_v2.json"
    if os.path.exists(summary_file_v2):
        with open(summary_file_v2) as f:
            summary_v2 = json.load(f)
        
        print(f"\nTotal gaming channels: {summary_v2.get('total_gaming_channels', 'N/A')}")
        print(f"Total non-gaming channels: {summary_v2.get('total_non_gaming_channels', 'N/A')}")
        print(f"Sample percentage: {summary_v2.get('sample_percentage', 'N/A')*100:.0f}%")
        print(f"Quantiles: {summary_v2.get('n_quantiles', 'N/A')}")
        
        print(f"\n{'Experiment':<30} {'Seeds':>8} {'Gaming':>8} {'Non-Gaming':>10} {'%':>8}")
        print("-"*70)
        for exp in summary_v2.get('experiments', []):
            pct = exp.get('actual_percentage', 0) * 100
            print(f"{exp['experiment_name']:<30} {exp.get('total_seeds', 0):>8} {exp.get('gaming_seeds', 0):>8} {exp.get('non_gaming_seeds', 0):>10} {pct:>7.2f}%")
    else:
        print(f"\n[INFO] {summary_file_v2} not found (v2 seeds not created)")
    
    # ============================================================
    # 3. GAMING CHANNELS DETAILS
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
        for q, stats in gaming.get('forward_count_by_quantile', {}).items():
            print(f"{q:<10} {stats['count']:>8} {stats['min_forward']:>10} {stats['max_forward']:>10} {stats['mean_forward']:>10.1f}")
    else:
        print(f"\n[INFO] {gaming_file} not found")
    
    # ============================================================
    # 4. EXPERIMENT RESULTS
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
                'tp': gt.get('true_positives', 0),
                'fp': gt.get('false_positives', 0),
                'fn': gt.get('false_negatives', 0),
                'precision': gt.get('precision', 0),
                'recall': gt.get('recall', 0),
                'f1': gt.get('f1', 0),
            })
    
    if results:
        print(f"\n{'Experiment':<30} {'Lvl':>4} {'Nodes':>7} {'Gaming':>7} {'TP':>5} {'FP':>5} {'FN':>6} {'Prec':>7} {'Rec':>7} {'F1':>7}")
        print("-"*100)
        for r in results:
            print(f"{r['experiment']:<30} {r['levels']:>4} {r['nodes']:>7} {r['gaming']:>7} {r['tp']:>5} {r['fp']:>5} {r['fn']:>6} {r['precision']*100:>6.1f}% {r['recall']*100:>6.1f}% {r['f1']*100:>6.1f}%")
        
        # Save to CSV
        df = pd.DataFrame(results)
        csv_file = f"{BASE_DIR}/results_comparison.csv"
        df.to_csv(csv_file, index=False)
        print(f"\nSaved to: {csv_file}")
    else:
        print("\n[INFO] No completed experiments found. Run analyze_all_levels.py first.")
    
    # ============================================================
    # 5. METRICS EXPLANATION
    # ============================================================
    print("\n" + "="*70)
    print(" METRICS EXPLANATION")
    print("="*70)
    print("""
Definizione metriche (canali senza etichetta ESCLUSI):

  TP (True Positives):
    Canali con etichetta "Videogame modding" E classificati come gaming
    
  FP (False Positives):
    Canali con etichetta ALTRO E classificati come gaming
    
  FN (False Negatives):
    Canali con etichetta "Videogame modding" E NON classificati
    (include TUTTI i GT gaming non catturati, anche quelli mai visitati)
    
  Precision = TP / (TP + FP)
    "Dei canali etichettati che classifico gaming, quanti sono davvero gaming?"
    
  Recall = TP / (TP + FN)  
    "Di tutti i canali 'Videogame modding' nel dataset, quanti ne ho catturati?"
    
  Nota: I canali predetti gaming ma SENZA etichetta sono riportati separatamente
        come 'Unlabeled' e non influenzano Precision/Recall.
""")

if __name__ == "__main__":
    main()