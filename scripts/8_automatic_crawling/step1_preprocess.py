#!/usr/bin/env python3
"""
STEP 1: Preprocess Telegram messages for topic detection.
Usage: python step1_preprocess.py --level 0

Output: preprocessing/
"""

import os
import re
import time
import argparse
import gc
from typing import Callable
from multiprocessing import Pool, cpu_count

import pandas as pd
from glob import glob
from tqdm import tqdm
from unidecode import unidecode
import langdetect
from spacy.lang.en.stop_words import STOP_WORDS

# ======================== TIMING UTILITIES ========================
START_TIME = time.perf_counter()

def log_time(message: str) -> None:
    elapsed = time.perf_counter() - START_TIME
    print(f"[{elapsed:8.2f}s] {message}")

# ======================== CONFIGURATION ========================
MIN_TOKENS_FOR_VALID_MESSAGE = 5
STOPWORDS = list(STOP_WORDS)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== PREPROCESSING CLASS ========================
class TextPreprocessor:
    def __init__(self, stopwords: list = None):
        self.stopwords = stopwords or []
        self.punctuation_pattern = r'[!"#$%&\'()*+,\-./:;<=>?@\\^_{|}~]'
    
    def _apply(self, text: str, func: Callable) -> str:
        if isinstance(text, str):
            return func(text)
        return ''
    
    def lowercase_and_normalize(self, text: str) -> str:
        text = self._apply(text, str.lower)
        return self._apply(text, unidecode)
    
    def remove_urls(self, text: str) -> str:
        return self._apply(text, lambda t: re.sub(r'http\S+', '', t).strip())
    
    def remove_mentions_hashtags(self, text: str) -> str:
        return self._apply(text, lambda t: re.sub(r'(@|#)\S+', '', t).strip())
    
    def remove_punctuation(self, text: str) -> str:
        text = self._apply(text, lambda t: re.sub(self.punctuation_pattern, ' ', t))
        text = self._apply(text, lambda t: re.sub(r'[\r\n]+', ' ', t))
        return self._apply(text, lambda t: re.sub(r' {2,}', ' ', t).strip())
    
    def remove_stopwords(self, text: str) -> str:
        if not self.stopwords:
            return text
        pattern = rf'\b({"|".join(self.stopwords)})\b'
        return self._apply(text, lambda t: re.sub(pattern, '', t).strip())
    
    def remove_short_words(self, text: str, min_length: int = 3) -> str:
        pattern = rf'(\b|^)\w{{1,{min_length}}}(\b|$)'
        return self._apply(text, lambda t: re.sub(pattern, '', t).strip())
    
    def remove_numbers(self, text: str) -> str:
        return self._apply(text, lambda t: re.sub(r'[0-9]+', '', t))
    
    def detect_language(self, text: str) -> str:
        try:
            detections = langdetect.detect_langs(text)
            best = max(detections, key=lambda x: x.prob)
            return best.lang if best.prob >= 0.7 else 'unk'
        except:
            return 'unk'

# ======================== PREPROCESSING FUNCTIONS ========================
def preprocess_for_lda(text: str) -> tuple:
    try:
        pp = TextPreprocessor(stopwords=STOPWORDS)
        text_normalized = pp.lowercase_and_normalize(text)
        lang = pp.detect_language(text_normalized)
        
        if lang in ('unk', None):
            return ("", "unk")
        
        text_clean = pp.remove_stopwords(text_normalized)
        text_clean = pp.remove_mentions_hashtags(text_clean)
        text_clean = pp.remove_urls(text_clean)
        text_clean = pp.remove_punctuation(text_clean)
        text_clean = pp.remove_numbers(text_clean)
        text_clean = pp.remove_short_words(text_clean, min_length=3)
        text_clean = ' '.join(text_clean.split())
        
        return (text_clean, lang)
    except:
        return ("", "unk")

def preprocess_for_llm(text: str) -> tuple:
    try:
        pp = TextPreprocessor()
        lang = pp.detect_language(text)
        
        if lang in ('unk', None):
            return ("", "unk")
        
        text_clean = pp.remove_urls(text)
        text_clean = ' '.join(text_clean.split())
        
        return (text_clean, lang)
    except:
        return ("", "unk")

# ======================== FILE PROCESSING ========================
def process_single_file(args: tuple) -> pd.DataFrame:
    filepath, channel_id = args
    try:
        df = pd.read_csv(filepath, sep='\t', compression='gzip', usecols=['text', 'timestamp'])
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].astype(str)
        
        lda_results = df['text'].apply(preprocess_for_lda)
        llm_results = df['text'].apply(preprocess_for_llm)
        
        df['text_lda'] = [r[0] for r in lda_results]
        df['text_llm'] = [r[0] for r in llm_results]
        df['language'] = [r[1] for r in lda_results]
        df['channel_id'] = channel_id
        
        return df if not df.empty else None
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def write_chunks(df: pd.DataFrame, path: str, chunk_size: int = 50000) -> None:
    for i, start in enumerate(range(0, len(df), chunk_size)):
        chunk = df.iloc[start:start+chunk_size]
        mode = 'w' if i == 0 else 'a'
        header = (i == 0)
        chunk.to_csv(path, sep='\t', index=False, header=header, 
                     mode=mode, compression='gzip')

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser(description="Preprocess Telegram messages")
    parser.add_argument("--level", type=str, default="0", help="Hierarchy level")
    args = parser.parse_args()
    
    level = args.level
    log_time(f"Starting preprocessing for level {level}")
    
    base_dir = f"../../results/levels_automatic/level_{level}"
    preprocess_dir = f"{base_dir}/preprocessing"
    os.makedirs(preprocess_dir, exist_ok=True)
    
    extracted_dir = '../../../../telegram_2024/usc-tg-24-us-election/extracted'
    nodes_file = f"{base_dir}/nodes_level_{level}.csv.gz"
    
    output_all_messages = f"{preprocess_dir}/messages_preprocessed.tsv.gz"
    output_english_clean = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    
    if os.path.exists(output_english_clean):
        log_time(f"Already processed: {output_english_clean}")
        return
    
    if not os.path.exists(nodes_file):
        log_time(f"ERROR: Nodes file not found: {nodes_file}")
        return
    
    df_nodes = pd.read_csv(nodes_file, compression="gzip")
    log_time(f"Loaded {len(df_nodes)} nodes from {nodes_file}")
    
    file_args = []
    for _, row in df_nodes.iterrows():
        channel_id = row['type_and_id']
        channel_path = os.path.join(extracted_dir, channel_id)
        files = glob(os.path.join(channel_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz'))
        
        if os.path.isdir(channel_path) and files:
            file_args.extend([(f, channel_id) for f in files])
    
    log_time(f"Found {len(file_args)} files to process")
    
    results = []
    with Pool(cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(process_single_file, file_args), 
                          total=len(file_args), desc="Processing"):
            if result is not None:
                results.append(result)
    
    if not results:
        log_time("ERROR: No messages processed")
        return
    
    df_all = pd.concat(results, ignore_index=True)
    log_time(f"Combined {len(df_all)} messages")
    del results
    gc.collect()
    
    df_all = df_all.dropna()
    df_all = df_all[df_all['text_lda'].str.strip() != '']
    df_all = df_all[df_all['text_llm'].str.strip() != '']
    
    df_english = df_all[df_all['language'] == 'en'].copy()
    log_time(f"English messages: {len(df_english)}")
    
    len_before = len(df_english)
    df_english = df_english.drop_duplicates(subset=['text_lda'])
    log_time(f"After dedup: {len(df_english)} (removed {len_before - len(df_english)})")
    
    df_english = df_english[
        df_english['text_lda'].str.split().apply(len) > MIN_TOKENS_FOR_VALID_MESSAGE
    ]
    log_time(f"After removing short messages: {len(df_english)}")
    
    write_chunks(df_all, output_all_messages)
    log_time(f"Saved all messages to {output_all_messages}")
    
    df_english.to_csv(output_english_clean, sep='\t', index=False, compression='gzip')
    log_time(f"Saved clean English messages to {output_english_clean}")
    
    total_time = time.perf_counter() - START_TIME
    log_time(f"COMPLETED in {total_time:.2f}s")
    
    with open(f"{preprocess_dir}/step1_completed.txt", "w") as f:
        f.write(f"Preprocessing completed in {total_time:.2f}s\n")
        f.write(f"Total messages: {len(df_all)}\n")
        f.write(f"English clean messages: {len(df_english)}\n")

if __name__ == "__main__":
    main()