#!/usr/bin/env python3
"""
VERIFICATION SCRIPT: Extract snippets from political and non-political channels for manual review.

Usage: 
    python verify_non_political_channels.py --max-level 5
    python verify_non_political_channels.py --max-level 5 --limit-political 50 --limit-non-political 100
    python verify_non_political_channels.py --max-level 5 --num-snippets 5 --snippet-messages 10

Output: 
    results/levels_automatic/verification/all_political_channels.json
    results/levels_automatic/verification/all_non_political_channels.json
    results/levels_automatic/verification/political_snippets.json
    results/levels_automatic/verification/non_political_snippets.json
"""

import os
import time
import argparse
import json
import pandas as pd
from glob import glob
from datetime import datetime

# ======================== TIMING ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== CONFIGURATION ========================
EXTRACTED_DIR = '../../../../telegram_2024/usc-tg-24-us-election/extracted'

# ======================== FUNCTIONS ========================
def get_channel_snippets(channel_id: str, num_snippets: int, snippet_messages: int) -> list:
    """
    Estrae snippet di conversazione da un canale, ordinati per timestamp.
    Ogni snippet contiene `snippet_messages` messaggi consecutivi.
    
    Args:
        channel_id: ID del canale
        num_snippets: numero di snippet da estrarre (distribuiti: inizio, metà, fine, ...)
        snippet_messages: numero di messaggi consecutivi per ogni snippet
    
    Returns:
        Lista di snippet, ognuno con lista di messaggi
    """
    channel_path = os.path.join(EXTRACTED_DIR, channel_id)
    
    if not os.path.isdir(channel_path):
        return []
    
    # Trova tutti i file mensili
    files = sorted(glob(os.path.join(channel_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz')))
    
    if not files:
        return []
    
    all_messages = []
    
    for filepath in files:
        try:
            df = pd.read_csv(filepath, sep='\t', compression='gzip', usecols=['text', 'timestamp'])
            df = df.dropna(subset=['text', 'timestamp'])
            df['text'] = df['text'].astype(str)
            all_messages.append(df)
        except Exception:
            continue
    
    if not all_messages:
        return []
    
    df_all = pd.concat(all_messages, ignore_index=True)
    
    # Ordina per timestamp
    df_all = df_all.sort_values('timestamp').reset_index(drop=True)
    
    n = len(df_all)
    if n == 0:
        return []
    
    # Calcola gli indici di inizio per ogni snippet (distribuiti uniformemente)
    if n <= snippet_messages:
        # Caso: meno messaggi di quelli richiesti per uno snippet
        # Restituisci un solo snippet con tutti i messaggi
        start_indices = [0]
    elif n <= num_snippets * snippet_messages:
        # Caso: abbastanza messaggi per alcuni snippet ma non tutti separati
        # Distribuisci il più possibile senza overlap
        available_snippets = max(1, n // snippet_messages)
        actual_snippets = min(num_snippets, available_snippets)
        if actual_snippets == 1:
            start_indices = [0]
        else:
            step = (n - snippet_messages) // (actual_snippets - 1)
            start_indices = [i * step for i in range(actual_snippets)]
    else:
        # Caso normale: abbastanza messaggi per tutti gli snippet
        if num_snippets == 1:
            start_indices = [n // 2]  # Centro
        elif num_snippets == 2:
            start_indices = [0, n - snippet_messages]
        elif num_snippets == 3:
            start_indices = [0, (n - snippet_messages) // 2, n - snippet_messages]
        else:
            # Distribuisci uniformemente
            step = (n - snippet_messages) // (num_snippets - 1)
            start_indices = [i * step for i in range(num_snippets)]
    
    snippets = []
    for start_idx in start_indices:
        # Assicurati che non vada oltre
        start_idx = min(start_idx, max(0, n - snippet_messages))
        end_idx = min(start_idx + snippet_messages, n)
        
        snippet_df = df_all.iloc[start_idx:end_idx]
        
        messages = []
        for _, row in snippet_df.iterrows():
            # Converti timestamp in formato leggibile
            try:
                ts = datetime.fromtimestamp(row['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
            except:
                ts = str(row['timestamp'])
            
            messages.append({
                "timestamp": ts,
                "text": row['text']
            })
        
        if messages:
            snippets.append({
                "start_index": int(start_idx),
                "end_index": int(end_idx),
                "messages": messages
            })
    
    return snippets

def collect_channels_from_levels(base_dir: str, max_level: int) -> tuple:
    """
    Raccoglie tutti i canali politici e non politici da tutti i livelli.
    """
    all_political = set()
    all_non_political = set()
    all_processed = set()
    
    for level in range(max_level + 1):
        political_path = f"{base_dir}/level_{level}/channel_analysis/political_channels.json"
        nodes_path = f"{base_dir}/level_{level}/nodes_level_{level}.csv.gz"
        
        # Canali processati in questo livello
        if os.path.exists(nodes_path):
            try:
                df_nodes = pd.read_csv(nodes_path, compression='gzip')
                if 'type_and_id' in df_nodes.columns:
                    all_processed.update(df_nodes['type_and_id'].tolist())
                    log_time(f"Level {level}: {len(df_nodes)} nodes processed")
            except:
                pass
        
        # Canali politici/non-politici classificati
        if os.path.exists(political_path):
            try:
                with open(political_path, 'r') as f:
                    data = json.load(f)
                
                political = set(data.get('political_channels', []))
                non_political = set(data.get('non_political_channels', []))
                
                all_political.update(political)
                all_non_political.update(non_political)
                
                log_time(f"Level {level}: {len(political)} political, {len(non_political)} non-political")
            except:
                pass
    
    return all_political, all_non_political, all_processed

def extract_snippets_for_channels(channel_list: list, num_snippets: int, snippet_messages: int, 
                                   label: str) -> dict:
    """
    Estrae snippet per una lista di canali.
    """
    snippets_dict = {}
    processed_count = 0
    error_count = 0
    
    for channel_id in channel_list:
        snippets = get_channel_snippets(channel_id, num_snippets, snippet_messages)
        
        if snippets:
            snippets_dict[channel_id] = {
                "num_snippets": len(snippets),
                "snippets": snippets
            }
            processed_count += 1
        else:
            error_count += 1
        
        total = processed_count + error_count
        if total % 50 == 0:
            log_time(f"  [{label}] Processed {total}/{len(channel_list)} channels...")
    
    log_time(f"  [{label}] Extracted snippets from {processed_count} channels ({error_count} errors)")
    
    return snippets_dict, processed_count, error_count

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Verify political and non-political channels with snippets")
    parser.add_argument("--max-level", type=int, default=10, 
                        help="Maximum level to check (default: 10)")
    parser.add_argument("--limit-political", type=int, default=None, 
                        help="Max number of political channels to sample (default: all)")
    parser.add_argument("--limit-non-political", type=int, default=None, 
                        help="Max number of non-political channels to sample (default: all)")
    parser.add_argument("--num-snippets", type=int, default=3, 
                        help="Number of snippets per channel (default: 3)")
    parser.add_argument("--snippet-messages", type=int, default=5, 
                        help="Number of consecutive messages per snippet (default: 5)")
    args = parser.parse_args()
    
    log_time("Starting verification of channels")
    log_time(f"Config: num_snippets={args.num_snippets}, snippet_messages={args.snippet_messages}")
    
    base_dir = "../../results/levels_automatic"
    verification_dir = f"{base_dir}/verification"
    os.makedirs(verification_dir, exist_ok=True)
    
    # Raccogli canali da tutti i livelli
    log_time("\nCollecting channels from all levels...")
    all_political, all_non_political, all_processed = collect_channels_from_levels(base_dir, args.max_level)
    
    log_time(f"\n{'='*60}")
    log_time(f"SUMMARY:")
    log_time(f"  Total processed channels: {len(all_processed)}")
    log_time(f"  Political channels: {len(all_political)}")
    log_time(f"  Non-political channels: {len(all_non_political)}")
    log_time(f"{'='*60}\n")
    
    # ==================== SALVA LISTE COMPLETE ====================
    
    # Salva lista canali politici
    political_summary = {
        "total_political": len(all_political),
        "political_channels": sorted(all_political)
    }
    political_summary_path = f"{verification_dir}/all_political_channels.json"
    with open(political_summary_path, 'w') as f:
        json.dump(political_summary, f, indent=2)
    log_time(f"Saved political channels list to {political_summary_path}")
    
    # Salva lista canali non-politici
    non_political_summary = {
        "total_non_political": len(all_non_political),
        "non_political_channels": sorted(all_non_political)
    }
    non_political_summary_path = f"{verification_dir}/all_non_political_channels.json"
    with open(non_political_summary_path, 'w') as f:
        json.dump(non_political_summary, f, indent=2)
    log_time(f"Saved non-political channels list to {non_political_summary_path}")
    
    # ==================== ESTRAI SNIPPET POLITICI ====================
    
    log_time("\n--- Extracting snippets from POLITICAL channels ---")
    political_list = sorted(all_political)
    if args.limit_political:
        political_list = political_list[:args.limit_political]
        log_time(f"Limited to {args.limit_political} political channels")
    else:
        log_time(f"Processing all {len(political_list)} political channels")
    
    if political_list:
        political_snippets, pol_ok, pol_err = extract_snippets_for_channels(
            political_list, args.num_snippets, args.snippet_messages, "POLITICAL"
        )
        
        political_output = {
            "metadata": {
                "total_political": len(all_political),
                "channels_sampled": len(political_list),
                "channels_with_snippets": pol_ok,
                "channels_without_data": pol_err,
                "num_snippets_per_channel": args.num_snippets,
                "messages_per_snippet": args.snippet_messages
            },
            "channels": political_snippets
        }
        
        political_snippets_path = f"{verification_dir}/political_snippets.json"
        with open(political_snippets_path, 'w') as f:
            json.dump(political_output, f, indent=2, ensure_ascii=False)
        log_time(f"Saved political snippets to {political_snippets_path}")
    else:
        log_time("No political channels to process")
    
    # ==================== ESTRAI SNIPPET NON-POLITICI ====================
    
    log_time("\n--- Extracting snippets from NON-POLITICAL channels ---")
    non_political_list = sorted(all_non_political)
    if args.limit_non_political:
        non_political_list = non_political_list[:args.limit_non_political]
        log_time(f"Limited to {args.limit_non_political} non-political channels")
    else:
        log_time(f"Processing all {len(non_political_list)} non-political channels")
    
    if non_political_list:
        non_political_snippets, nonpol_ok, nonpol_err = extract_snippets_for_channels(
            non_political_list, args.num_snippets, args.snippet_messages, "NON-POLITICAL"
        )
        
        non_political_output = {
            "metadata": {
                "total_non_political": len(all_non_political),
                "channels_sampled": len(non_political_list),
                "channels_with_snippets": nonpol_ok,
                "channels_without_data": nonpol_err,
                "num_snippets_per_channel": args.num_snippets,
                "messages_per_snippet": args.snippet_messages
            },
            "channels": non_political_snippets
        }
        
        non_political_snippets_path = f"{verification_dir}/non_political_snippets.json"
        with open(non_political_snippets_path, 'w') as f:
            json.dump(non_political_output, f, indent=2, ensure_ascii=False)
        log_time(f"Saved non-political snippets to {non_political_snippets_path}")
    else:
        log_time("No non-political channels to process")
    
    # ==================== STAMPA ESEMPI ====================
    
    log_time(f"\n{'='*60}")
    log_time("SAMPLE POLITICAL CHANNELS:")
    log_time(f"{'='*60}")
    
    if political_list and political_snippets:
        sample_political = list(political_snippets.keys())[:3]
        for channel_id in sample_political:
            log_time(f"\n🔴 {channel_id}:")
            channel_data = political_snippets[channel_id]
            for s_idx, snippet in enumerate(channel_data['snippets'][:2], 1):  # Max 2 snippet per esempio
                log_time(f"   Snippet {s_idx} (msgs {snippet['start_index']}-{snippet['end_index']}):")
                for msg in snippet['messages'][:3]:  # Max 3 messaggi per snippet
                    text_preview = msg['text'][:80].replace('\n', ' ')
                    if len(msg['text']) > 80:
                        text_preview += "..."
                    log_time(f"      [{msg['timestamp']}] {text_preview}")
    
    log_time(f"\n{'='*60}")
    log_time("SAMPLE NON-POLITICAL CHANNELS:")
    log_time(f"{'='*60}")
    
    if non_political_list and non_political_snippets:
        sample_non_political = list(non_political_snippets.keys())[:3]
        for channel_id in sample_non_political:
            log_time(f"\n🟢 {channel_id}:")
            channel_data = non_political_snippets[channel_id]
            for s_idx, snippet in enumerate(channel_data['snippets'][:2], 1):
                log_time(f"   Snippet {s_idx} (msgs {snippet['start_index']}-{snippet['end_index']}):")
                for msg in snippet['messages'][:3]:
                    text_preview = msg['text'][:80].replace('\n', ' ')
                    if len(msg['text']) > 80:
                        text_preview += "..."
                    log_time(f"      [{msg['timestamp']}] {text_preview}")
    
    # ==================== FINAL SUMMARY ====================
    
    total_time = time.perf_counter() - START_TIME
    log_time(f"\n{'='*60}")
    log_time(f"COMPLETED in {total_time:.2f}s")
    log_time(f"{'='*60}")
    log_time(f"Output files:")
    log_time(f"  - {verification_dir}/all_political_channels.json")
    log_time(f"  - {verification_dir}/all_non_political_channels.json")
    log_time(f"  - {verification_dir}/political_snippets.json")
    log_time(f"  - {verification_dir}/non_political_snippets.json")

if __name__ == "__main__":
    main()