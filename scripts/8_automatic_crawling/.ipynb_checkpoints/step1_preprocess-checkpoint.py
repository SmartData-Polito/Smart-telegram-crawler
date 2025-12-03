#!/usr/bin/env python3
"""
STEP 1: Preprocess Telegram messages for topic detection.
Usage: python step1_preprocess.py --level 0

Output: preprocessing/

OTTIMIZZAZIONI:
- Language detection fatta una sola volta per messaggio
- Pattern regex pre-compilati
- Preprocessing unificato
"""

import os
import re
import time
import argparse
import gc
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
STOPWORDS = set(STOP_WORDS)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ======================== PRE-COMPILED PATTERNS ========================
PATTERN_URL = re.compile(r'http\S+')
PATTERN_MENTIONS = re.compile(r'(@|#)\S+')
PATTERN_PUNCTUATION = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@\\^_{|}~]')
PATTERN_NEWLINES = re.compile(r'[\r\n]+')
PATTERN_SPACES = re.compile(r' {2,}')
PATTERN_NUMBERS = re.compile(r'[0-9]+')
PATTERN_SHORT_WORDS = re.compile(r'\b\w{1,3}\b')
PATTERN_STOPWORDS = re.compile(rf'\b({"|".join(STOPWORDS)})\b')

# ======================== LANGUAGE DETECTION ========================
def detect_language(text: str) -> str:
    """Detect language con langdetect."""
    if not text or len(text.strip()) < 20:
        return 'unk'
    try:
        # Limita testo per velocità
        text_sample = text[:300]
        detections = langdetect.detect_langs(text_sample)
        best = max(detections, key=lambda x: x.prob)
        return best.lang if best.prob >= 0.7 else 'unk'
    except:
        return 'unk'

# ======================== PREPROCESSING FUNCTIONS ========================
def preprocess_text(text: str) -> dict:
    """
    Preprocessa un testo una sola volta, restituisce sia versione LDA che LLM.
    """
    if not isinstance(text, str) or not text.strip():
        return {"text_lda": "", "text_llm": "", "language": "unk"}
    
    # 1. Detecta lingua UNA SOLA VOLTA
    lang = detect_language(text)
    
    if lang == 'unk':
        return {"text_lda": "", "text_llm": "", "language": "unk"}
    
    # 2. Versione LLM (pulizia leggera)
    text_llm = PATTERN_URL.sub('', text)
    text_llm = ' '.join(text_llm.split())
    
    # 3. Versione LDA (pulizia pesante)
    text_lda = unidecode(text.lower())
    text_lda = PATTERN_STOPWORDS.sub('', text_lda)
    text_lda = PATTERN_MENTIONS.sub('', text_lda)
    text_lda = PATTERN_URL.sub('', text_lda)
    text_lda = PATTERN_PUNCTUATION.sub(' ', text_lda)
    text_lda = PATTERN_NEWLINES.sub(' ', text_lda)
    text_lda = PATTERN_NUMBERS.sub('', text_lda)
    text_lda = PATTERN_SHORT_WORDS.sub('', text_lda)
    text_lda = PATTERN_SPACES.sub(' ', text_lda).strip()
    
    return {"text_lda": text_lda, "text_llm": text_llm, "language": lang}

# ======================== FILE PROCESSING ========================
def process_single_file(args: tuple) -> pd.DataFrame:
    """Processa un singolo file."""
    filepath, channel_id = args
    
    try:
        df = pd.read_csv(filepath, sep='\t', compression='gzip', usecols=['text', 'timestamp'])
        df = df.dropna(subset=['text'])
        
        if df.empty:
            return None
        
        df['text'] = df['text'].astype(str)
        
        # Processa tutti i testi
        results = [preprocess_text(t) for t in df['text']]
        
        df['text_lda'] = [r['text_lda'] for r in results]
        df['text_llm'] = [r['text_llm'] for r in results]
        df['language'] = [r['language'] for r in results]
        df['channel_id'] = channel_id
        
        return df
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}")
        return None

def write_chunks(df: pd.DataFrame, path: str, chunk_size: int = 50000) -> None:
    """Scrive DataFrame in chunks."""
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
    
    log_time("Concatenating results...")
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
    
    log_time("Saving all messages...")
    write_chunks(df_all, output_all_messages)
    log_time(f"Saved all messages to {output_all_messages}")
    
    log_time("Saving English messages...")
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