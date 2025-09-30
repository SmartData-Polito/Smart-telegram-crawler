#!/usr/bin/env python3
import argparse, os, sys, gzip
import pandas as pd
import numpy as np
from pathlib import Path

# python 1_troubleShooting.py --input 0 --short-threshold 3 --spam-threshold 3
# python 1_troubleShooting.py --input 0 --grid-file ../../results/levels/level_0/grid_search/df_sampled_level_0.csv

def exists_nonempty(path: str) -> bool:
    p = Path(path)
    return p.exists() and p.stat().st_size > 0

def safe_read_tsv(path: str, expected_cols=None, usecols=None, msg=""):
    if not exists_nonempty(path):
        print(f"[ERRORE] File mancante o vuoto: {path}")
        return None
    try:
        df = pd.read_csv(path, sep='\t', compression='gzip', usecols=usecols)
    except Exception as e:
        print(f"[ERRORE] Lettura fallita: {path} -> {type(e).__name__}: {e}")
        return None
    if expected_cols:
        missing = [c for c in expected_cols if c not in df.columns]
        if missing:
            print(f"[WARN] Colonne mancanti in {msg or path}: {missing}")
    return df

def summarize_basic(df: pd.DataFrame, name: str):
    print("\n" + "="*80)
    print(f">>> {name}")
    print(df.head())
    print(f"len: {len(df)}")
    # NaN per colonna (prime 10)
    nan_counts = df.isna().sum().sort_values(ascending=False)
    if (nan_counts > 0).any():
        print("[INFO] NaN per colonna (top 10):")
        print(nan_counts.head(10))

def ensure_string_clean(series: pd.Series, min_len=1):
    s = series.astype('string')
    s = s.str.replace('\u00A0', ' ', regex=False)                      # NBSP
    s = s.str.replace(r'[\u200B-\u200D\uFEFF]', '', regex=True)        # zero-width
    s = s.str.replace(r'\s+', ' ', regex=True).str.strip()             # spazi multipli
    s = s.replace({'nan': pd.NA, 'null': pd.NA, 'None': pd.NA})
    s = s.where(s.str.len() >= min_len, other=pd.NA)
    return s

def token_len(series: pd.Series):
    return series.astype('string').str.split().str.len()

def check_short_df(short_df: pd.DataFrame, main_df: pd.DataFrame, short_threshold: int):
    if 'text_preprocessed' not in short_df.columns or 'text_preprocessed' not in main_df.columns:
        print("[WARN] Impossibile validare short_df: manca 'text_preprocessed'.")
        return
    # Ricomputa token len sul main e conta quante righe avrebbero dovuto essere short
    main_tokens = token_len(main_df['text_preprocessed'])
    expected_short = (main_tokens <= short_threshold).sum()
    print(f"[CHECK] short_df.rows = {len(short_df)} ; attese (<= {short_threshold} tokens) = {expected_short}")

def check_spam_df(spam_df: pd.DataFrame, spam_threshold: int):
    needed = {'channel_id', 'text_preprocessed', 'count_spam'}
    if not needed.issubset(spam_df.columns):
        print(f"[WARN] Impossibile validare spam_df: richiede colonne {needed}.")
        return
    ok = (spam_df['count_spam'] > spam_threshold).all()
    print(f"[CHECK] spam_df rispetta soglia > {spam_threshold}: {bool(ok)}")
    if not ok:
        bad = spam_df.loc[spam_df['count_spam'] <= spam_threshold].head(5)
        print("[ESEMPIO] Righe che non rispettano la soglia:\n", bad)

def check_channels_without(df_without: pd.DataFrame, df_eng: pd.DataFrame):
    if 'channel_id' not in df_without.columns:
        print("[WARN] channels_without_message non ha 'channel_id'.")
        return
    seen = set(df_eng['channel_id'].unique()) if 'channel_id' in df_eng.columns else set()
    missing_list = df_without['channel_id'].tolist()
    inter = [c for c in missing_list if c in seen]
    if inter:
        print(f"[WARN] {len(inter)} canali risultano 'senza messaggi' ma compaiono nel dataset EN. Esempi: {inter[:5]}")
    else:
        print("[CHECK] channels_without_message coerente con i canali visti in EN.")

def print_text_stats(df: pd.DataFrame, col='text_preprocessed', topn=10):
    if col not in df.columns:
        print(f"[WARN] Nessuna colonna '{col}' in dataframe.")
        return
    s = ensure_string_clean(df[col])
    tok = token_len(s)
    print(f"[STAT] {col}:")
    print(f"  - Vuoti/NA: {(s.isna()).sum()} / {len(s)}")
    print(f"  - Lunghezza token: min={tok.min()} max={tok.max()} mediana={tok.median()} media={tok.mean():.2f}")
    # duplicati
    dup = s.duplicated().sum()
    print(f"  - Duplicati esatti: {dup} ({dup/len(s)*100:.2f}%)")
    # top duplicati
    vc = s.value_counts()
    if len(vc) > 0:
        print(f"  - Top {topn} testi duplicati:")
        print(vc.head(topn))

def print_lang_stats(df: pd.DataFrame):
    if 'language' in df.columns:
        vc = df['language'].astype('string').value_counts(dropna=False)
        print("[STAT] Distribuzione lingua:")
        print(vc.head(10))

def print_time_stats(df: pd.DataFrame):
    if 'timestamp' in df.columns:
        ts = pd.to_numeric(df['timestamp'], errors='coerce')
        if ts.notna().any():
            print(f"[STAT] timestamp range: [{int(ts.min())} .. {int(ts.max())}]")
        else:
            print("[WARN] Timestamp non numerici/assenti.")

def compare_with_grid_file(grid_file: str, df_ref: pd.DataFrame):
    if not grid_file:
        return
    if not exists_nonempty(grid_file):
        print(f"[WARN] grid_file non trovato: {grid_file}")
        return
    try:
        df_grid = pd.read_csv(grid_file)
    except Exception as e:
        print(f"[WARN] impossibile leggere grid_file: {type(e).__name__}: {e}")
        return
    print("\n" + "-"*80)
    print(f"[CHECK] Confronto con grid_file: {grid_file}")
    print(f"  - righe grid_file: {len(df_grid)}")
    # verifica che text_preprocessed del grid_file stia dentro al ref
    if 'text_preprocessed' in df_grid.columns and 'text_preprocessed' in df_ref.columns:
        ref_set = set(df_ref['text_preprocessed'].astype('string').unique())
        grid_set = set(df_grid['text_preprocessed'].astype('string').unique())
        missing = len(grid_set - ref_set)
        print(f"  - voci in grid_file non presenti nel ref: {missing}")
    else:
        print("  - impossibile confrontare: manca 'text_preprocessed' in uno dei due.")

def main():
    ap = argparse.ArgumentParser(description="Stampa risultati finali e controlli robusti sui dataframe di preprocessing.")
    ap.add_argument("--input", type=str, default="0", help="Level depth (default: 0)")
    ap.add_argument("--short-threshold", type=int, default=3, help="Soglia token per short_df (<= soglia). Default: 3")
    ap.add_argument("--spam-threshold", type=int, default=3, help="Soglia count_spam per spam_df (> soglia). Default: 3")
    ap.add_argument("--grid-file", type=str, default="", help="(Opzionale) CSV usato in grid_search (df_sampled) per confronto coerenza")
    args = ap.parse_args()

    level_depth = args.input
    base_dir = f"../../results/levels/level_{level_depth}/preProcessing/"
    path_non_empty_eng = os.path.join(base_dir, f"preprocessed_english_messages_level_{level_depth}.tsv.gz")
    path_non_empty_eng_no_dupes_short = os.path.join(base_dir, f"preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz")
    path_short = os.path.join(base_dir, f"preprocessed_short_messages_level_{level_depth}.tsv.gz")
    path_spam = os.path.join(base_dir, f"preprocessed_spam_messages_level_{level_depth}.tsv.gz")
    path_channels_without_msg = os.path.join(base_dir, f"channels_without_message_level_{level_depth}.tsv.gz")

    # Carico con controlli base
    df_eng_full = safe_read_tsv(
        path_non_empty_eng,
        expected_cols=['text', 'text_preprocessed', 'language', 'channel_id', 'timestamp'],
        msg='preprocessed_english_messages'
    )
    df_no_dupes_short = safe_read_tsv(
        path_non_empty_eng_no_dupes_short,
        expected_cols=['text', 'text_preprocessed', 'language', 'channel_id', 'timestamp'],
        msg='preprocessed_non_empty_english_channels_without_duplicates_and_short_messages'
    )
    short_df = safe_read_tsv(
        path_short,
        expected_cols=['text_preprocessed', 'channel_id'],
        msg='preprocessed_short_messages'
    )
    spam_df = safe_read_tsv(
        path_spam,
        expected_cols=['channel_id', 'text_preprocessed', 'count_spam'],
        msg='preprocessed_spam_messages'
    )
    df_channels_without = safe_read_tsv(
        path_channels_without_msg,
        expected_cols=['channel_id'],
        msg='channels_without_message'
    )

    if df_eng_full is None or df_no_dupes_short is None or short_df is None or spam_df is None or df_channels_without is None:
        print("[FATALE] Impossibile procedere: uno o più dataframe non sono stati caricati.")
        sys.exit(2)

    # Stampo le anteprime
    summarize_basic(df_no_dupes_short, "df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages")
    summarize_basic(df_channels_without, "df_channels_without_messages")
    summarize_basic(short_df, "short_df (short messages)")
    summarize_basic(spam_df, "spam_df (spam messages)")

    # Statistiche testo/lingua/tempo
    print_text_stats(df_no_dupes_short, 'text_preprocessed')
    print_lang_stats(df_no_dupes_short)
    print_time_stats(df_no_dupes_short)

    # Controlli coerenza short/spam/channels
    check_short_df(short_df, df_eng_full, args.short_threshold)
    check_spam_df(spam_df, args.spam_threshold)
    check_channels_without(df_channels_without, df_eng_full)

    # Coerenza: il file usato in grid_search (df_sampled) rispetto al ref (qui uso il “no_dupes_short”)
    compare_with_grid_file(args.grid_file, df_no_dupes_short)

    print("\n" + "="*80)
    print(">>> Controlli completati.")
    print("="*80 + "\n")

if __name__ == "__main__":
    main()
