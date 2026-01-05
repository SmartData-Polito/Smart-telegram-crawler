#!/usr/bin/env python3
"""
STEP 1: Preprocess Telegram messages for topic detection.
Usage: python step1_preprocess.py --level 0
       python step1_preprocess.py --level 0 --base-dir ../../results/experiments/peak_jul_aug
       python step1_preprocess.py --level 0 --start-date 2024-07-15 --end-date 2024-08-15
"""

import os
import re
import time
import argparse
import json
import gc
from multiprocessing import Pool, cpu_count
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
from glob import glob
from tqdm import tqdm
from unidecode import unidecode

# ======================== TIMING ========================
START_TIME = time.perf_counter()
STEP_TIMES = {}

def log_time(msg: str) -> None:
    print(f"[{time.perf_counter() - START_TIME:8.2f}s] {msg}")

def start_timer(name: str) -> float:
    return time.perf_counter()

def end_timer(name: str, start: float) -> float:
    elapsed = time.perf_counter() - start
    STEP_TIMES[name] = elapsed
    return elapsed

# ======================== CONFIG ========================
MIN_TOKENS = 5
SAVE_EVERY = 500
FILTER_CHUNKSIZE = 500000  # Increased for better performance

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"

# ======================== FASTTEXT LANGUAGE DETECTION ========================
FASTTEXT_MODEL = None

def init_fasttext():
    global FASTTEXT_MODEL
    try:
        import fasttext
        fasttext.FastText.eprint = lambda x: None
        
        model_paths = [
            '/tmp/lid.176.ftz',
            os.path.expanduser('~/lid.176.ftz'),
            'lid.176.ftz'
        ]
        
        model_path = None
        for p in model_paths:
            if os.path.exists(p):
                model_path = p
                break
        
        if not model_path:
            import urllib.request
            model_path = '/tmp/lid.176.ftz'
            log_time("Downloading fasttext model...")
            urllib.request.urlretrieve(
                'https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz',
                model_path
            )
        
        FASTTEXT_MODEL = fasttext.load_model(model_path)
        log_time("FastText model loaded")
        return True
    except Exception as e:
        log_time(f"FastText not available: {e}, using fallback")
        return False

def detect_lang_fast(text: str) -> str:
    if not text or len(text) < 15:
        return 'unk'
    try:
        pred = FASTTEXT_MODEL.predict(text[:150].replace('\n', ' '), k=1)
        lang = pred[0][0].replace('__label__', '')
        conf = pred[1][0]
        return lang if conf > 0.5 else 'unk'
    except:
        return 'unk'

def detect_lang_fallback(text: str) -> str:
    if not text or len(text) < 20:
        return 'unk'
    try:
        import langdetect
        return langdetect.detect(text[:150])
    except:
        return 'unk'

# ======================== REGEX ========================
PATTERN_CLEAN_ALL = re.compile(
    r'https?://\S+|www\.\S+|'
    r'[@#]\S+|'
    r'[!"#$%&\'()*+,\-./:;<=>?@\[\]\\^_`{|}~]|'
    r'\d+|'
    r'[\r\n]+',
    re.IGNORECASE
)

PATTERN_SHORT = re.compile(r'\b\w{1,3}\b')
PATTERN_SPACES = re.compile(r'\s+')

from spacy.lang.en.stop_words import STOP_WORDS
STOPWORDS = frozenset(STOP_WORDS)

def remove_stopwords_fast(text: str) -> str:
    return ' '.join(w for w in text.split() if w not in STOPWORDS)

# ======================== PREPROCESSING ========================
def preprocess_text(text: str) -> tuple:
    if not text or not isinstance(text, str) or len(text) < 10:
        return ('', '', 'unk')
    
    detect_fn = detect_lang_fast if FASTTEXT_MODEL else detect_lang_fallback
    lang = detect_fn(text)
    if lang == 'unk':
        return ('', '', 'unk')
    
    text_llm = PATTERN_SPACES.sub(' ', text.replace('\n', ' ')).strip()
    if text_llm.startswith(('http://', 'https://')):
        text_llm = ''
    
    text_lda = unidecode(text.lower())
    text_lda = PATTERN_CLEAN_ALL.sub(' ', text_lda)
    text_lda = remove_stopwords_fast(text_lda)
    text_lda = PATTERN_SHORT.sub('', text_lda)
    text_lda = PATTERN_SPACES.sub(' ', text_lda).strip()
    
    return (text_lda, text_llm, lang)

def parse_timestamp(ts) -> datetime:
    """Parse timestamp from various formats."""
    if pd.isna(ts):
        return None
    
    if isinstance(ts, datetime):
        return ts
    
    if isinstance(ts, (int, float)):
        try:
            return datetime.fromtimestamp(ts)
        except:
            return None
    
    if isinstance(ts, str):
        formats = [
            '%Y-%m-%d %H:%M:%S',
            '%Y-%m-%dT%H:%M:%S',
            '%Y-%m-%d',
            '%d/%m/%Y %H:%M:%S',
            '%d/%m/%Y',
        ]
        for fmt in formats:
            try:
                return datetime.strptime(ts[:19], fmt)
            except:
                continue
        
        try:
            return pd.to_datetime(ts).to_pydatetime()
        except:
            return None
    
    return None

# ======================== FILE PROCESSING ========================
def process_single_file(args: tuple) -> dict:
    filepath, channel_id, start_date, end_date = args
    
    try:
        df = pd.read_csv(filepath, sep='\t', compression='gzip', 
                        usecols=['text', 'timestamp'],
                        dtype={'text': str})
        df = df.dropna(subset=['text'])
        
        if df.empty:
            return {'data': None, 'channel_id': channel_id, 'error': None, 
                    'total_before_filter': 0, 'total_after_filter': 0}
        
        total_before_filter = len(df)
        
        # Apply date filter if specified
        if start_date is not None or end_date is not None:
            df['parsed_timestamp'] = df['timestamp'].apply(parse_timestamp)
            
            if start_date is not None:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                df = df[df['parsed_timestamp'] >= start_dt]
            
            if end_date is not None:
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                end_dt = end_dt.replace(hour=23, minute=59, second=59)
                df = df[df['parsed_timestamp'] <= end_dt]
            
            df = df.drop(columns=['parsed_timestamp'])
        
        total_after_filter = len(df)
        
        if df.empty:
            return {'data': None, 'channel_id': channel_id, 'error': None,
                    'total_before_filter': total_before_filter, 'total_after_filter': 0}
        
        texts = df['text'].tolist()
        timestamps = df['timestamp'].tolist()
        
        results = [preprocess_text(t) for t in texts]
        
        data = {
            'text': texts,
            'timestamp': timestamps,
            'text_lda': [r[0] for r in results],
            'text_llm': [r[1] for r in results],
            'language': [r[2] for r in results],
            'channel_id': [channel_id] * len(texts)
        }
        
        return {'data': data, 'channel_id': channel_id, 'error': None,
                'total_before_filter': total_before_filter, 'total_after_filter': total_after_filter}
        
    except Exception as e:
        return {'data': None, 'channel_id': channel_id, 'error': str(e),
                'total_before_filter': 0, 'total_after_filter': 0}

# ======================== MAIN ========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=str, default="0")
    parser.add_argument("--base-dir", type=str, default="../../results/levels_automatic",
                        help="Base directory for results")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--start-date", type=str, default=None,
                        help="Start date filter (YYYY-MM-DD)")
    parser.add_argument("--end-date", type=str, default=None,
                        help="End date filter (YYYY-MM-DD)")
    args = parser.parse_args()
    
    level = args.level
    base_dir = args.base_dir
    n_workers = args.workers or min(16, max(1, cpu_count() - 2))
    start_date = args.start_date
    end_date = args.end_date
    
    log_time(f"Starting preprocessing level {level} with {n_workers} workers")
    log_time(f"  Base dir: {base_dir}")
    if start_date:
        log_time(f"  Start date filter: {start_date}")
    if end_date:
        log_time(f"  End date filter: {end_date}")
    
    # Init fasttext
    t_start = start_timer("init_fasttext")
    use_fasttext = init_fasttext()
    end_timer("init_fasttext", t_start)
    
    # Paths
    level_dir = f"{base_dir}/level_{level}"
    preprocess_dir = f"{level_dir}/preprocessing"
    os.makedirs(preprocess_dir, exist_ok=True)
    
    extracted_dir = '../../../../telegram_2024/usc-tg-24-us-election/extracted'
    nodes_file = f"{level_dir}/nodes_level_{level}.csv.gz"
    
    output_english = f"{preprocess_dir}/messages_english_clean.tsv.gz"
    output_tracking = f"{preprocess_dir}/channels_tracking.json"
    temp_file = f"{preprocess_dir}/_temp_all.tsv"
    
    if os.path.exists(output_english) and os.path.exists(output_tracking):
        log_time(f"Already done: {output_english}")
        return
    
    if not os.path.exists(nodes_file):
        log_time(f"ERROR: {nodes_file} not found")
        return
    
    # Load nodes
    t_start = start_timer("load_nodes")
    df_nodes = pd.read_csv(nodes_file, compression="gzip")
    all_nodes = set(df_nodes['type_and_id'].tolist())
    end_timer("load_nodes", t_start)
    log_time(f"Loaded {len(all_nodes)} nodes")
    
    # Scan files
    t_start = start_timer("scan_files")
    no_folder, no_files, with_files = [], [], {}
    file_args = []
    
    for ch in df_nodes['type_and_id']:
        ch_path = os.path.join(extracted_dir, ch)
        if not os.path.isdir(ch_path):
            no_folder.append(ch)
            continue
        
        files = glob(os.path.join(ch_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz'))
        if not files:
            no_files.append(ch)
            continue
        
        with_files[ch] = len(files)
        file_args.extend([(f, ch, start_date, end_date) for f in files])
    
    end_timer("scan_files", t_start)
    log_time(f"Files: {len(file_args)} from {len(with_files)} channels")
    log_time(f"Skipped: {len(no_folder)} no folder, {len(no_files)} no files")
    
    # Handle no files case
    if not file_args:
        log_time("WARNING: No files to process - creating empty outputs")
        
        df_empty = pd.DataFrame(columns=['text', 'timestamp', 'text_lda', 'text_llm', 'language', 'channel_id'])
        df_empty.to_csv(output_english, sep='\t', index=False, compression='gzip')
        
        tracking = {
            "level": level,
            "start_date": start_date,
            "end_date": end_date,
            "total_nodes": len(all_nodes),
            "total_messages": 0,
            "total_messages_before_date_filter": 0,
            "english_messages": 0,
            "summary": {
                "channels_no_folder": len(no_folder),
                "channels_no_files": len(no_files),
                "channels_lost_in_processing": 0,
                "channels_no_english": 0,
                "channels_only_short_messages": 0,
                "channels_with_valid_messages": 0
            },
            "details": {
                "channels_no_folder": sorted(no_folder),
                "channels_no_files": sorted(no_files),
                "channels_lost_in_processing": [],
                "channels_no_english": [],
                "channels_only_short_messages": [],
                "channels_with_valid_messages": []
            }
        }
        
        with open(output_tracking, 'w') as f:
            json.dump(tracking, f, indent=2)
        
        total_time = time.perf_counter() - START_TIME
        STEP_TIMES["total"] = total_time
        
        with open(f"{preprocess_dir}/step1_completed.txt", 'w') as f:
            f.write(f"Step 1: Preprocessing\n")
            f.write(f"Level: {level}\n")
            f.write(f"Base dir: {base_dir}\n")
            f.write(f"Status: COMPLETED (no files)\n")
            f.write(f"Date filter: {start_date} to {end_date}\n")
            f.write(f"Total time: {total_time:.2f}s\n\n")
            f.write(f"Timing breakdown:\n")
            for step_name, step_time in STEP_TIMES.items():
                f.write(f"  {step_name}: {step_time:.2f}s\n")
        return
    
    # Process files
    t_start = start_timer("process_files")
    channels_processed = set()
    all_errors = []
    total_msgs = 0
    total_msgs_before_filter = 0
    first_write = True
    
    buffer = {k: [] for k in ['text', 'timestamp', 'text_lda', 'text_llm', 'language', 'channel_id']}
    buffer_count = 0
    
    def flush_buffer():
        nonlocal buffer, buffer_count, first_write, total_msgs
        if not buffer['text']:
            return
        
        df_buf = pd.DataFrame(buffer)
        total_msgs += len(df_buf)
        
        if first_write:
            df_buf.to_csv(temp_file, sep='\t', index=False, mode='w')
            first_write = False
        else:
            df_buf.to_csv(temp_file, sep='\t', index=False, mode='a', header=False)
        
        buffer = {k: [] for k in buffer.keys()}
        buffer_count = 0
        gc.collect()
    
    log_time("Processing files with incremental save...")
    
    with Pool(n_workers) as pool:
        for result in tqdm(pool.imap_unordered(process_single_file, file_args),
                          total=len(file_args), desc="Processing"):
            if result['error']:
                all_errors.append((result['channel_id'], result['error']))
                continue
            
            total_msgs_before_filter += result.get('total_before_filter', 0)
            
            if result['data']:
                channels_processed.add(result['channel_id'])
                for k in buffer:
                    buffer[k].extend(result['data'][k])
                buffer_count += 1
                
                if buffer_count >= SAVE_EVERY:
                    flush_buffer()
    
    flush_buffer()
    end_timer("process_files", t_start)
    log_time(f"Total messages before date filter: {total_msgs_before_filter}")
    log_time(f"Total messages after date filter: {total_msgs}")
    
    # Handle no messages case
    if total_msgs == 0 or not os.path.exists(temp_file):
        log_time("WARNING: No messages processed - creating empty outputs")
        
        df_empty = pd.DataFrame(columns=['text', 'timestamp', 'text_lda', 'text_llm', 'language', 'channel_id'])
        df_empty.to_csv(output_english, sep='\t', index=False, compression='gzip')
        
        ch_lost = set(with_files.keys()) - channels_processed
        
        tracking = {
            "level": level,
            "start_date": start_date,
            "end_date": end_date,
            "total_nodes": len(all_nodes),
            "total_messages": 0,
            "total_messages_before_date_filter": total_msgs_before_filter,
            "english_messages": 0,
            "summary": {
                "channels_no_folder": len(no_folder),
                "channels_no_files": len(no_files),
                "channels_lost_in_processing": len(ch_lost),
                "channels_no_english": 0,
                "channels_only_short_messages": 0,
                "channels_with_valid_messages": 0
            },
            "details": {
                "channels_no_folder": sorted(no_folder),
                "channels_no_files": sorted(no_files),
                "channels_lost_in_processing": sorted(ch_lost),
                "channels_no_english": [],
                "channels_only_short_messages": [],
                "channels_with_valid_messages": []
            }
        }
        
        with open(output_tracking, 'w') as f:
            json.dump(tracking, f, indent=2)
        
        total_time = time.perf_counter() - START_TIME
        STEP_TIMES["total"] = total_time
        
        with open(f"{preprocess_dir}/step1_completed.txt", 'w') as f:
            f.write(f"Step 1: Preprocessing\n")
            f.write(f"Level: {level}\n")
            f.write(f"Base dir: {base_dir}\n")
            f.write(f"Status: COMPLETED (no messages)\n")
            f.write(f"Date filter: {start_date} to {end_date}\n")
            f.write(f"Total time: {total_time:.2f}s\n\n")
            f.write(f"Timing breakdown:\n")
            for step_name, step_time in STEP_TIMES.items():
                f.write(f"  {step_name}: {step_time:.2f}s\n")
        return
    
    # =================================================================
    # SKIP saving all messages - NOT NEEDED by subsequent steps!
    # Only messages_english_clean.tsv.gz is used by step2+
    # =================================================================
    t_start = start_timer("skip_full_save")
    log_time("Skipping full messages save (only English file needed by pipeline)")
    end_timer("skip_full_save", t_start)
    
    # Filter English directly from temp file
    t_start = start_timer("filter_english")
    log_time(f"Filtering English messages (chunksize={FILTER_CHUNKSIZE})...")
    
    chunk_iter = pd.read_csv(temp_file, sep='\t', chunksize=FILTER_CHUNKSIZE)
    english_chunks = []
    ch_with_english = set()
    
    for chunk in tqdm(chunk_iter, desc="Filtering"):
        mask = (
            (chunk['language'] == 'en') &
            (chunk['text_lda'].astype(str).str.len() > 0) &
            (chunk['text_llm'].astype(str).str.len() > 0)
        )
        en_chunk = chunk[mask]
        if len(en_chunk) > 0:
            ch_with_english.update(en_chunk['channel_id'].unique())
            english_chunks.append(en_chunk)
    
    # Remove temp file immediately after reading
    if os.path.exists(temp_file):
        os.remove(temp_file)
        log_time("Removed temp file")
    
    end_timer("filter_english", t_start)
    
    # Handle no English case
    if not english_chunks:
        log_time("WARNING: No English messages found - creating empty file")
        
        df_en = pd.DataFrame(columns=['text', 'timestamp', 'text_lda', 'text_llm', 'language', 'channel_id'])
        df_en.to_csv(output_english, sep='\t', index=False, compression='gzip')
        
        ch_lost = set(with_files.keys()) - channels_processed
        ch_no_en = channels_processed
        
        tracking = {
            "level": level,
            "start_date": start_date,
            "end_date": end_date,
            "total_nodes": len(all_nodes),
            "total_messages": total_msgs,
            "total_messages_before_date_filter": total_msgs_before_filter,
            "english_messages": 0,
            "summary": {
                "channels_no_folder": len(no_folder),
                "channels_no_files": len(no_files),
                "channels_lost_in_processing": len(ch_lost),
                "channels_no_english": len(ch_no_en),
                "channels_only_short_messages": 0,
                "channels_with_valid_messages": 0
            },
            "details": {
                "channels_no_folder": sorted(no_folder),
                "channels_no_files": sorted(no_files),
                "channels_lost_in_processing": sorted(ch_lost),
                "channels_no_english": sorted(ch_no_en),
                "channels_only_short_messages": [],
                "channels_with_valid_messages": []
            }
        }
        
        with open(output_tracking, 'w') as f:
            json.dump(tracking, f, indent=2)
        
        total_time = time.perf_counter() - START_TIME
        STEP_TIMES["total"] = total_time
        
        with open(f"{preprocess_dir}/step1_completed.txt", 'w') as f:
            f.write(f"Step 1: Preprocessing\n")
            f.write(f"Level: {level}\n")
            f.write(f"Base dir: {base_dir}\n")
            f.write(f"Status: COMPLETED (no English)\n")
            f.write(f"Date filter: {start_date} to {end_date}\n")
            f.write(f"Total time: {total_time:.2f}s\n\n")
            f.write(f"Timing breakdown:\n")
            for step_name, step_time in STEP_TIMES.items():
                f.write(f"  {step_name}: {step_time:.2f}s\n")
        return
    
    # Concatenate English
    t_start = start_timer("concat_english")
    log_time("Concatenating English chunks...")
    df_en = pd.concat(english_chunks, ignore_index=True)
    del english_chunks
    gc.collect()
    end_timer("concat_english", t_start)
    log_time(f"English messages: {len(df_en)}")
    
    # Dedup
    t_start = start_timer("dedup")
    len_before = len(df_en)
    df_en = df_en.drop_duplicates(subset=['text_lda'])
    end_timer("dedup", t_start)
    log_time(f"After dedup: {len(df_en)} (-{len_before - len(df_en)})")
    
    ch_before_short = set(df_en['channel_id'].unique())
    
    # Remove short
    t_start = start_timer("filter_short")
    token_counts = df_en['text_lda'].str.split().str.len()
    df_en = df_en[token_counts > MIN_TOKENS]
    end_timer("filter_short", t_start)
    log_time(f"After short filter: {len(df_en)}")
    
    ch_final = set(df_en['channel_id'].unique())
    
    # Save English
    t_start = start_timer("save_english")
    df_en.to_csv(output_english, sep='\t', index=False, compression='gzip')
    end_timer("save_english", t_start)
    log_time(f"Saved: {output_english}")
    
    # Tracking
    ch_lost = set(with_files.keys()) - channels_processed
    ch_no_en = channels_processed - ch_with_english
    ch_short = ch_before_short - ch_final
    
    log_time(f"\n{'='*50}")
    log_time(f"TRACKING: {len(all_nodes)} nodes")
    log_time(f"  No folder: {len(no_folder)}")
    log_time(f"  No files: {len(no_files)}")
    log_time(f"  Errors: {len(ch_lost)}")
    log_time(f"  No English: {len(ch_no_en)}")
    log_time(f"  Only short: {len(ch_short)}")
    log_time(f"  Final: {len(ch_final)}")
    log_time(f"{'='*50}")
    
    tracking = {
        "level": level,
        "start_date": start_date,
        "end_date": end_date,
        "total_nodes": len(all_nodes),
        "total_messages": total_msgs,
        "total_messages_before_date_filter": total_msgs_before_filter,
        "english_messages": len(df_en),
        "summary": {
            "channels_no_folder": len(no_folder),
            "channels_no_files": len(no_files),
            "channels_lost_in_processing": len(ch_lost),
            "channels_no_english": len(ch_no_en),
            "channels_only_short_messages": len(ch_short),
            "channels_with_valid_messages": len(ch_final)
        },
        "details": {
            "channels_no_folder": sorted(no_folder),
            "channels_no_files": sorted(no_files),
            "channels_lost_in_processing": sorted(ch_lost),
            "channels_no_english": sorted(ch_no_en),
            "channels_only_short_messages": sorted(ch_short),
            "channels_with_valid_messages": sorted(ch_final)
        }
    }
    
    with open(output_tracking, 'w') as f:
        json.dump(tracking, f, indent=2)
    
    # Final timing
    total_time = time.perf_counter() - START_TIME
    STEP_TIMES["total"] = total_time
    log_time(f"DONE in {total_time:.1f}s ({total_time/60:.1f} min)")
    
    with open(f"{preprocess_dir}/step1_completed.txt", 'w') as f:
        f.write(f"Step 1: Preprocessing\n")
        f.write(f"Level: {level}\n")
        f.write(f"Base dir: {base_dir}\n")
        f.write(f"Status: COMPLETED\n")
        f.write(f"Date filter: {start_date} to {end_date}\n")
        f.write(f"Total time: {total_time:.2f}s\n\n")
        f.write(f"Timing breakdown:\n")
        for step_name, step_time in STEP_TIMES.items():
            f.write(f"  {step_name}: {step_time:.2f}s\n")
        f.write(f"\nResults:\n")
        f.write(f"  Total messages before date filter: {total_msgs_before_filter}\n")
        f.write(f"  Total messages after date filter: {total_msgs}\n")
        f.write(f"  English clean: {len(df_en)}\n")
        f.write(f"  Channels: {len(ch_final)}\n")
        f.write(f"  Language detection: {'fasttext' if use_fasttext else 'langdetect'}\n")

if __name__ == "__main__":
    main()