#!/usr/bin/env python3
"""
Step 1: Preprocess messages from TGDataset JSON files.
Extracts English messages and saves them for LDA.
"""

import os
import sys
import json
import gzip
import tarfile
import argparse
import re
from datetime import datetime
from collections import defaultdict

import pandas as pd
import fasttext
from tqdm import tqdm

fasttext.FastText.eprint = lambda x: None

# ======================== CONFIGURATION ========================
FASTTEXT_MODEL_PATH = "../../material/lid.176.bin"
TGDATASET_DIR = "../../material"
MIN_MESSAGE_LENGTH = 15

# ======================== HELPER FUNCTIONS ========================
def load_fasttext_model():
    """Load FastText language detection model."""
    if not os.path.exists(FASTTEXT_MODEL_PATH):
        print(f"Downloading FastText model...")
        import urllib.request
        url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
        urllib.request.urlretrieve(url, FASTTEXT_MODEL_PATH)
    return fasttext.load_model(FASTTEXT_MODEL_PATH)

def detect_language(text, model):
    """Detect language of text using FastText."""
    if not text or len(text.strip()) < MIN_MESSAGE_LENGTH:
        return None, 0.0
    
    text_clean = text.replace('\n', ' ').strip()
    
    try:
        predictions = model.predict(text_clean, k=1)
        lang = predictions[0][0].replace('__label__', '')
        confidence = predictions[1][0]
        return lang, confidence
    except:
        return None, 0.0

def clean_text_for_lda(text):
    """Clean text for LDA processing."""
    if not text:
        return ""
    
    text = re.sub(r'http\S+|www\.\S+|t\.me/\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = ' '.join(text.split())
    
    return text.strip()

def load_channel_mapping():
    """Load channel to file mapping."""
    mapping_file = f"{TGDATASET_DIR}/channel_file_mapping.json"
    
    if not os.path.exists(mapping_file):
        raise FileNotFoundError(f"Channel mapping not found: {mapping_file}\nRun create_seeds.py first.")
    
    with open(mapping_file, 'r') as f:
        return json.load(f)

def load_channels_data(channel_ids, channel_mapping):
    """Load data for multiple channels efficiently."""
    results = {}
    
    # Group by tar file
    by_tar = defaultdict(list)
    for ch_id in channel_ids:
        ch_id_str = str(ch_id)
        if ch_id_str in channel_mapping:
            info = channel_mapping[ch_id_str]
            by_tar[info['tar']].append((ch_id_str, info['file']))
    
    for tar_name, items in by_tar.items():
        tar_path = f"{TGDATASET_DIR}/{tar_name}"
        
        # Group by file
        by_file = defaultdict(list)
        for ch_id, file_path in items:
            by_file[file_path].append(ch_id)
        
        print(f"  Reading from {tar_name}...")
        with tarfile.open(tar_path, 'r:gz') as tar:
            for file_path, ch_ids_in_file in tqdm(by_file.items(), desc="  Files"):
                try:
                    f = tar.extractfile(file_path)
                    if f:
                        data = json.load(f)
                        for ch_id in ch_ids_in_file:
                            if ch_id in data:
                                results[ch_id] = data[ch_id]
                except:
                    pass
    
    return results

# ======================== MAIN PROCESSING ========================
def process_level(level, base_dir, start_date=None, end_date=None):
    """Process all channels for a given level."""
    
    print(f"\n{'='*60}")
    print(f"STEP 1: PREPROCESSING - LEVEL {level}")
    print(f"{'='*60}")
    
    level_dir = f"{base_dir}/level_{level}"
    output_dir = f"{level_dir}/preprocessing"
    os.makedirs(output_dir, exist_ok=True)
    
    # Get channels to process
    nodes_file = f"{level_dir}/nodes_level_{level}.csv.gz"
    if not os.path.exists(nodes_file):
        print(f"[ERROR] Nodes file not found: {nodes_file}")
        return
    
    df_nodes = pd.read_csv(nodes_file, compression='gzip')
    channels = df_nodes['channel_id'].tolist()
    print(f"Channels to process: {len(channels)}")
    
    # Load models and mappings
    print("\nLoading FastText model...")
    ft_model = load_fasttext_model()
    
    print("Loading channel mapping...")
    channel_mapping = load_channel_mapping()
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d") if start_date else None
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") if end_date else None
    
    if start_dt:
        print(f"Date filter: {start_date} to {end_date}")
    
    # Load all channel data
    print("\nLoading channel data...")
    channels_data = load_channels_data(channels, channel_mapping)
    
    # Process channels
    print("\nProcessing messages...")
    all_messages = []
    tracking = {
        'total_nodes': len(channels),
        'channels_found': len(channels_data),
        'channels_not_found': len(channels) - len(channels_data),
        'channels_no_english': 0,
        'channels_with_english': 0,
        'total_messages': 0,
        'messages_in_date_range': 0,
        'english_messages': 0,
        'total_forwarded_messages': 0,
    }
    
    for ch_id_str, channel_data in tqdm(channels_data.items(), desc="Channels"):
        text_messages = channel_data.get('text_messages', {})
        tracking['total_messages'] += len(text_messages)
        
        channel_english_msgs = []
        
        for msg_id, msg in text_messages.items():
            msg_text = msg.get('message', '')
            msg_date = msg.get('date')
            forwarded_from_id = msg.get('forwarded_from_id')
            
            if forwarded_from_id is not None:
                tracking['total_forwarded_messages'] += 1
            
            if not msg_text or len(msg_text) < MIN_MESSAGE_LENGTH:
                continue
            
            # Date filter
            if msg_date and (start_dt or end_dt):
                try:
                    msg_datetime = datetime.fromtimestamp(msg_date)
                    if start_dt and msg_datetime < start_dt:
                        continue
                    if end_dt and msg_datetime > end_dt:
                        continue
                except:
                    continue
            
            tracking['messages_in_date_range'] += 1
            
            # Language detection
            lang, conf = detect_language(msg_text, ft_model)
            
            if lang == 'en' and conf > 0.5:
                cleaned = clean_text_for_lda(msg_text)
                if len(cleaned) >= MIN_MESSAGE_LENGTH:
                    channel_english_msgs.append({
                        'channel_id': int(ch_id_str),
                        'message_id': msg_id,
                        'text_lda': cleaned,              # Per LDA training
                        'text_llm': msg_text[:500],       # Testo originale per sample docs
                        'date': msg_date,
                        'forwarded_from_id': forwarded_from_id
                    })
        
        if channel_english_msgs:
            tracking['channels_with_english'] += 1
            tracking['english_messages'] += len(channel_english_msgs)
            all_messages.extend(channel_english_msgs)
        else:
            tracking['channels_no_english'] += 1
    
    # Save results
    print(f"\nSaving results...")
    
    if all_messages:
        df = pd.DataFrame(all_messages)
        output_file = f"{output_dir}/messages_english_clean.tsv.gz"
        df.to_csv(output_file, sep='\t', index=False, compression='gzip')
        print(f"  Saved: {output_file}")
        print(f"  Columns: {list(df.columns)}")
    else:
        print("  [WARN] No English messages found!")
    
    with open(f"{output_dir}/channels_tracking.json", 'w') as f:
        json.dump(tracking, f, indent=2)
    
    # Summary
    print(f"\n{'='*60}")
    print(f"PREPROCESSING SUMMARY")
    print(f"{'='*60}")
    print(f"  Total nodes:              {tracking['total_nodes']:,}")
    print(f"  Channels found:           {tracking['channels_found']:,}")
    print(f"  Channels not found:       {tracking['channels_not_found']:,}")
    print(f"  Channels with English:    {tracking['channels_with_english']:,}")
    print(f"  Channels no English:      {tracking['channels_no_english']:,}")
    print(f"  Total messages:           {tracking['total_messages']:,}")
    print(f"  Forwarded messages:       {tracking['total_forwarded_messages']:,}")
    print(f"  Messages in date range:   {tracking['messages_in_date_range']:,}")
    print(f"  English messages:         {tracking['english_messages']:,}")
    
    with open(f"{output_dir}/step1_completed.txt", 'w') as f:
        f.write(f"Completed at {datetime.now()}\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--level', type=str, required=True)
    parser.add_argument('--base-dir', type=str, required=True)
    parser.add_argument('--start-date', type=str, default=None)
    parser.add_argument('--end-date', type=str, default=None)
    args = parser.parse_args()
    
    process_level(args.level, args.base_dir, args.start_date, args.end_date)

if __name__ == "__main__":
    main()