#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script per calcolare il True Positive Rate (TPR) della classificazione machine
rispetto alla ground truth human.

TPR = TP / (TP + FN)
dove:
- TP (True Positives) = topic classificati come politics sia da human che da machine
- FN (False Negatives) = topic classificati come politics da human ma NON da machine
"""

import argparse

def calculate_tpr(human_politics, human_non_politics, machine_politics, machine_non_politics):
    """
    Calcola TPR e altre metriche di classificazione.
    
    Ground Truth: human
    Prediction: machine
    Positive Class: politics
    """
    
    # Conversione a set per operazioni efficienti
    human_politics_set = set(human_politics)
    human_non_politics_set = set(human_non_politics)
    machine_politics_set = set(machine_politics)
    machine_non_politics_set = set(machine_non_politics)
    
    # True Positives: classificati politics da entrambi
    TP = human_politics_set & machine_politics_set
    
    # False Negatives: politics per human, non-politics per machine
    FN = human_politics_set & machine_non_politics_set
    
    # False Positives: non-politics per human, politics per machine
    FP = human_non_politics_set & machine_politics_set
    
    # True Negatives: non-politics per entrambi
    TN = human_non_politics_set & machine_non_politics_set
    
    # Calcola TPR (Sensitivity/Recall per la classe positiva)
    tpr = len(TP) / (len(TP) + len(FN)) if (len(TP) + len(FN)) > 0 else 0.0
    
    # Metriche aggiuntive
    # Precision per politics
    precision = len(TP) / (len(TP) + len(FP)) if (len(TP) + len(FP)) > 0 else 0.0
    
    # F1-Score
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0
    
    # Accuracy
    accuracy = (len(TP) + len(TN)) / (len(TP) + len(TN) + len(FP) + len(FN))
    
    # Specificity (TNR - True Negative Rate)
    specificity = len(TN) / (len(TN) + len(FP)) if (len(TN) + len(FP)) > 0 else 0.0
    
    return {
        'TP': TP,
        'FN': FN,
        'FP': FP,
        'TN': TN,
        'TPR': tpr,
        'Precision': precision,
        'F1': f1,
        'Accuracy': accuracy,
        'Specificity': specificity,
        'n_TP': len(TP),
        'n_FN': len(FN),
        'n_FP': len(FP),
        'n_TN': len(TN)
    }


def print_results(results, verbose=False):
    """Stampa i risultati in formato leggibile."""
    
    print("=" * 80)
    print("TPR CALCULATION - Machine vs Human Ground Truth")
    print("=" * 80)
    print()
    
    print("CONFUSION MATRIX:")
    print("-" * 80)
    print(f"                    | Human: Politics | Human: Non-Politics |")
    print(f"Machine: Politics   |  TP = {results['n_TP']:3d}      |  FP = {results['n_FP']:3d}           |")
    print(f"Machine: Non-Pol.   |  FN = {results['n_FN']:3d}      |  TN = {results['n_TN']:3d}           |")
    print("-" * 80)
    print()
    
    print("METRICS:")
    print("-" * 80)
    print(f"TPR (Recall) TP/ (TP + FN):  {results['TPR']:.4f}  ({results['TPR']*100:.2f}%)")
    print(f"Precision TP / (TP / FP):                 {results['Precision']:.4f}  ({results['Precision']*100:.2f}%)")
    print(f"F1-Score:                  {results['F1']:.4f}  ({results['F1']*100:.2f}%)")
    print(f"Accuracy:                  {results['Accuracy']:.4f}  ({results['Accuracy']*100:.2f}%)")
    print(f"Specificity (TNR):         {results['Specificity']:.4f}  ({results['Specificity']*100:.2f}%)")
    print("-" * 80)
    print()
    
    print(f"Total topics: {results['n_TP'] + results['n_FN'] + results['n_FP'] + results['n_TN']}")
    print(f"Human Politics: {results['n_TP'] + results['n_FN']}")
    print(f"Machine Politics: {results['n_TP'] + results['n_FP']}")
    print()
    
    if verbose:
        print("DETAILED BREAKDOWN:")
        print("-" * 80)
        print(f"\nTrue Positives (n={results['n_TP']}):")
        print(f"  Topics: {sorted(results['TP'])}")
        print(f"\nFalse Negatives (n={results['n_FN']}):")
        print(f"  Topics: {sorted(results['FN'])}")
        print(f"  (Human said politics, Machine said non-politics)")
        print(f"\nFalse Positives (n={results['n_FP']}):")
        print(f"  Topics: {sorted(results['FP'])}")
        print(f"  (Human said non-politics, Machine said politics)")
        print(f"\nTrue Negatives (n={results['n_TN']}):")
        print(f"  Topics: {sorted(results['TN'])}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Calcola TPR della classificazione machine vs human ground truth"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Mostra dettagli completi dei topic classificati"
    )
    args = parser.parse_args()
    
    # Ground Truth (Human)
    human_politics_topics = [
        0, 3, 4, 6, 7, 8, 11, 12, 13, 14,
        15, 17, 18, 20, 21, 22, 23, 24, 26, 27,
        28, 30, 31, 32, 33, 34, 35, 38, 41, 43,
        45, 46, 48, 49, 50, 51, 52, 53, 54, 55,
        56, 57
    ]
    
    human_non_politics_topics = [
        1, 2, 5, 9, 10, 16, 19, 25, 29, 36,
        37, 39, 40, 42, 44, 47
    ]
    
    # Prediction (Machine)
    machine_politics_topics = [
        0, 3, 6, 7, 8, 11, 14, 15, 17, 18, 20, 21, 22, 23, 
        24, 26, 27, 28, 30, 31, 32, 33, 34, 37, 38, 41, 43,
        44, 45, 46, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57
    ]
    machine_non_politics_topics = [
        1, 2, 4, 5, 9, 10, 12, 13, 16, 19,
        25, 29, 35, 36, 39, 40, 42, 47
    ]
    
    # Calcola metriche
    results = calculate_tpr(
        human_politics_topics,
        human_non_politics_topics,
        machine_politics_topics,
        machine_non_politics_topics
    )
    
    # Stampa risultati
    print_results(results, verbose=args.verbose)


if __name__ == "__main__":
    main()