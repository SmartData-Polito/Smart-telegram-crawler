#!/usr/bin/env python3
import pandas as pd
import argparse
import os

# Parametri da linea di comando (stesso --input)
parser = argparse.ArgumentParser(description="Stampa i risultati finali dei preprocessing dataframe.")
parser.add_argument(
    "--input",
    type=str,
    default="0",
    help="Depth of the hierarchy (default: 0)"
)
args = parser.parse_args()
level_depth = args.input

# Path dei file di output generati dallo script principale
base_dir = f"../../results/levels/level_{level_depth}/preProcessing/"
path_non_empty_eng = os.path.join(base_dir, f"preprocessed_english_messages_level_{level_depth}.tsv.gz")
path_non_empty_eng_no_dupes_short = os.path.join(base_dir, f"preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz")
path_short = os.path.join(base_dir, f"preprocessed_short_messages_level_{level_depth}.tsv.gz")
path_spam = os.path.join(base_dir, f"preprocessed_spam_messages_level_{level_depth}.tsv.gz")
path_channels_without_msg = os.path.join(base_dir, f"channels_without_message_level_{level_depth}.tsv.gz")

# Carico i dataframe
df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages = pd.read_csv(path_non_empty_eng_no_dupes_short, sep='\t', compression='gzip')
df_channels_without_messages = pd.read_csv(path_channels_without_msg, sep='\t', compression='gzip')
short_df = pd.read_csv(path_short, sep='\t', compression='gzip')
spam_df = pd.read_csv(path_spam, sep='\t', compression='gzip')

# Stampo i risultati richiesti
print("\n" + "="*80)
print(">>> df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages")
print(df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages.head())
print("len:", len(df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages))

print("\n" + "="*80)
print(">>> df_channels_without_messages")
print(df_channels_without_messages.head())
print("len:", len(df_channels_without_messages))

print("\n" + "="*80)
print(">>> short_df (short messages)")
print(short_df.head())
print("len:", len(short_df))

print("\n" + "="*80)
print(">>> spam_df (spam messages)")
print(spam_df.head())
print("len:", len(spam_df))
print("="*80 + "\n")

