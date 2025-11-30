## Preprocessng of the texts

#imports
import os
import re
from typing import Callable, Union
from unidecode import unidecode
import langdetect
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import pandas as pd
from glob import glob
import argparse
from spacy.lang.en.stop_words import STOP_WORDS
import gc

# Start preprocessing
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#getting parameter from command line
parser = argparse.ArgumentParser(description="Preprocess messages for a list of Telegram channels.")
parser.add_argument(
    "--input",
    type=str,
    default="0",
    help="Depth of the hierarchy(default: 0)"
)
args = parser.parse_args()
level_depth = args.input

# Create directory
level_dir = f"../results/levels/level_{level_depth}/preProcessing/"
os.makedirs(level_dir, exist_ok=True)

# Input path
extracted_dir = '../../../telegram_2024/usc-tg-24-us-election/extracted'
input_path_df_political_nodes = os.path.join(level_dir, f"nodes_level_{level_depth}.csv.gz")

# Read input list
df_nodes = pd.read_csv(input_path_df_political_nodes)
print(df_nodes.head())