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

# Output paths
output_path_preprocessed_messages = os.path.join(level_dir, f"preprocessed_messages_level_{level_depth}.tsv.gz")
output_path_preprocessed_english_messages = os.path.join(level_dir, f"preprocessed_english_messages_level_{level_depth}.tsv.gz")
output_path_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages = os.path.join(
    level_dir,
    f"preprocessed_non_empty_english_channels_without_duplicates_and_short_messages_level_{level_depth}.tsv.gz"
)
output_path_preprocessed_non_empty_english_channels_without_short_messages = os.path.join(
    level_dir,
    f"preprocessed_non_empty_english_channels_without_short_messages_level_{level_depth}.tsv.gz"
)
output_path_preprocessed_messages_only_with_short_messages = os.path.join(level_dir, f"preprocessed_short_messages_level_{level_depth}.tsv.gz")
output_path_preprocessed_messages_only_with_spam_messages = os.path.join(level_dir, f"preprocessed_spam_messages_level_{level_depth}.tsv.gz")
output_path_channels_without_message = os.path.join(level_dir, f"channels_without_message_level_{level_depth}.tsv.gz")


#variables
considered_short_tokens = 5
considered_spam_threshold = 6

#Preprocessing class
class PreProcessing:
    def __init__(self, noadverbs: bool = False, noadjectives: bool = False, noverbs: bool = False,
                 noentities: bool = False, language: str = 'en', remove_list: bool = False, stopwords=[]):
        """Initialize the PreProcessing object.

        Args:
            noadverbs (bool, optional): Flag to indicate whether to remove adverbs. Defaults to False.
            noadjectives (bool, optional): Flag to indicate whether to remove adjectives. Defaults to False.
            noverbs (bool, optional): Flag to indicate whether to remove verbs. Defaults to False.
            noentities (bool, optional): Flag to indicate whether to remove named entities. Defaults to False.
            remove_list (bool, optional): Flag to indicate whether to remove stopwords. Defaults to False.
        """
        self.noadverbs = noadverbs
        self.noadjectives = noadjectives
        self.noverbs = noverbs
        self.noentities = noentities
        self.remove_list = remove_list
        self.punctuation = r'[!"#$%&\'()*+,\-./:;<=>?@\[\\\]^_`{|}~]'
        self.stopwords = stopwords

    def _process_text(self, text: Union[str, list], function: Callable) -> Union[str, list]:
        if isinstance(text, str):
            return function(text)
        elif isinstance(text, list):
            return [function(x) for x in text]
        return ''

    def lowercase_unidecode(self, text: Union[str, list]) -> Union[str, list]:
        text = self._process_text(text, lambda value: value.lower())
        text = self._process_text(text, unidecode)
        return text

    def remove_urls(self, text):
        return self._process_text(text, lambda value: re.sub(r'http\S+ *', '', value).strip())

    def remove_tweet_marking(self, text):
        return self._process_text(text, lambda value: re.sub(r'(@|#)\S+ *', '', value).strip())

    def remove_html_tags(self, text):
        return self._process_text(text, lambda value: re.sub(r'<.*?> *', '', value).strip())

    def remove_punctuation(self, text):
        text = self._process_text(text, lambda value: re.sub(self.punctuation, ' ', value))
        text = self._process_text(text, lambda value: re.sub(r'[\r\n]+', ' ', value))
        text = self._process_text(text, lambda value: re.sub(' {2,}', ' ', value).strip())
        return text

    def remove_repetition(self, text):
        return self._process_text(text, lambda value: re.sub(r'\b(\w+)\s+\1\b', r'\1', value))

    def append_stopwords_list(self, stopwords: list):
        self.stopwords.extend(stopwords)

    def remove_stopwords(self, text):
        return self._process_text(text, lambda value: re.sub(rf'\b({"|".join(self.stopwords)})\b *', '', value).strip())

    def remove_n(self, text, n: int):
        return self._process_text(text, lambda value: re.sub(rf'(\b|^)\w{{1,{n}}}(\b|$) ?', '', value).strip())

    def remove_numbers(self, text, mode: str = 'replace'):
        if mode == "filter":
            return self._process_text(text, lambda value: '' if re.search('[0-9]', value) else value)
        elif mode == "replace":
            return self._process_text(text, lambda value: re.sub('[0-9] *', '', value))

    def remove_gerund(self, text):
        return self._process_text(text, lambda value: re.sub(r'ndo\b', '', value))

    def remove_infinitive(self, text):
        return self._process_text(text, lambda value: re.sub(r'r\b', '', value))

    def detect_language(self, text):
        try:
            d = langdetect.detect_langs(text)
            langs_dict = {lang.lang: lang.prob for lang in d}
            best_lang = max(langs_dict, key=langs_dict.get)
            best_lang = best_lang if langs_dict[best_lang] >= 0.7 else 'unk'
            return best_lang
        except langdetect.LangDetectException:
            return 'unk'


#preprocess_text function
stopwords = list(STOP_WORDS)
def preprocess_text(text, stopwords=stopwords):
    try:
        pp = PreProcessing(language='en', stopwords=stopwords)
        text_low = pp.lowercase_unidecode(text)
        lang = pp.detect_language(text_low)
        if lang in ('unk', None):
            return ("", "unk")
        text_clean = pp.remove_stopwords(text_low)
        text_clean = pp.remove_tweet_marking(text_clean)
        text_clean = pp.remove_urls(text_clean)
        text_clean = pp.remove_repetition(text_clean)
        text_clean = pp.remove_punctuation(text_clean)
        text_clean = pp.remove_numbers(text_clean)
        text_clean = pp.remove_n(text_clean, n=3)
        return (text_clean, lang)
    except Exception:
        return ("", "unk")

#process_file function definition and return dataframw with channel_id, text, text_preprocessed, language
def process_file(args):
    file, channel_id = args
    try:
        df = pd.read_csv(file, sep='\t', compression='gzip', usecols=['text', 'timestamp'])
        df = df.dropna(subset=['text'])
        df['text'] = df['text'].astype(str)
        pairs = df['text'].apply(preprocess_text)
        df['text_preprocessed'] = [p[0] for p in pairs]
        df['language'] = [p[1] for p in pairs]
        df = df[df['text_preprocessed'] != ""]
        if df.empty:
            return None
        df['channel_id'] = channel_id
        return df
    except Exception as e:
        print(f"--- Error in file {file}: {type(e).__name__}: {e}")
        return None

#write_df_in_chunks function definition
def write_df_in_chunks(df, path, sep='\t', chunk_size=50000):
    first = True
    for i in range(0, len(df), chunk_size):
        df.iloc[i:i+chunk_size].to_csv(
            path,
            sep=sep,
            index=False,
            header=first,
            mode='w' if first else 'a',
            compression='gzip'
        )
        first = False

# Read input list
df_first_nodes = pd.read_csv(input_path_df_political_nodes)
print(df_first_nodes.head())

if os.path.exists(output_path_preprocessed_messages):
    print("--- File already exists: {}".format(output_path_preprocessed_messages))
    df_preprocessed_non_empty_channels = pd_read = pd.read_csv(output_path_preprocessed_messages, sep='\t', compression='gzip')
    print("--- File loaded with {} preprocessed messages.".format(len(df_preprocessed_non_empty_channels)))
else:
    #interate from extracted for every file with channels_id
    file_args = []
    for _, row in df_first_nodes.iterrows():
        channel_id = row['type_and_id']
        channel_path = os.path.join(extracted_dir, channel_id)
        files = glob(os.path.join(channel_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz'))
        if not os.path.isdir(channel_path) or not files:
            continue
        file_args.extend([(file, channel_id) for file in files])

    #multiprocessing
    dfs = []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_file, file_args), total=len(file_args)):
            if res is not None:
                dfs.append(res)
    if dfs:
        df_preprocessed_non_empty_channels = pd.concat(dfs, ignore_index=True)
    else:
        df_preprocessed_non_empty_channels = pd.DataFrame()

# DROP NaN, ALL STRINGS
df_preprocessed_non_empty_channels=df_preprocessed_non_empty_channels.dropna()
df_preprocessed_non_empty_channels.loc[:, 'text'] = (
    df_preprocessed_non_empty_channels['text']
      .astype('string')
      .str.strip()
)
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels[
    df_preprocessed_non_empty_channels['text'] != ""
]
df_preprocessed_non_empty_channels=df_preprocessed_non_empty_channels.dropna()
df_preprocessed_non_empty_channels.loc[:, 'text_preprocessed'] = (
    df_preprocessed_non_empty_channels['text_preprocessed']
      .astype('string')
      .str.strip()
)
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels[
    df_preprocessed_non_empty_channels['text_preprocessed'] != ""
]

# DATAFRAME OF ENGLISH ONLY MESSAGES
df_preprocessed_non_empty_english_channels = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['language'] == 'en']
#write df_preprocessed_non_empty_channels to file
write_df_in_chunks(df_preprocessed_non_empty_channels, output_path_preprocessed_messages)

# DATAFRAME WITHOUT SHORT MESSAGES
df_preprocessed_non_empty_channels_without_short = df_preprocessed_non_empty_channels.copy()
df_preprocessed_non_empty_channels_without_short = df_preprocessed_non_empty_channels_without_short[df_preprocessed_non_empty_channels_without_short['text_preprocessed'].str.split().apply(len) > considered_short_tokens]
write_df_in_chunks(df_preprocessed_non_empty_channels_without_short, output_path_preprocessed_non_empty_english_channels_without_short_messages)
del df_preprocessed_non_empty_channels_without_short
gc.collect()
del df_preprocessed_non_empty_channels
gc.collect()

#DROPPING DUPLICATES
print("len before drop_duplicates:", len(df_preprocessed_non_empty_english_channels))
df_preprocessed_non_empty_english_channels_without_duplicates = df_preprocessed_non_empty_english_channels.drop_duplicates(subset=['text_preprocessed'])
print("len after drop_duplicates:", len(df_preprocessed_non_empty_english_channels_without_duplicates))

#DROPPING SHORT MESSAGES
print("len before dropping short message:", len(df_preprocessed_non_empty_english_channels_without_duplicates))
df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages = df_preprocessed_non_empty_english_channels_without_duplicates[df_preprocessed_non_empty_english_channels_without_duplicates['text_preprocessed'].str.split().apply(len) > considered_short_tokens]
print("len after dropping duplicates and short messages:", len(df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages))
#write df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages
df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages.to_csv(output_path_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages, sep='\t', index=False, compression='gzip')
del df_preprocessed_non_empty_english_channels_without_duplicates_and_short_messages
gc.collect()

#SHORTS MESSAGES DATAFRAME
short_df = df_preprocessed_non_empty_english_channels_without_duplicates[df_preprocessed_non_empty_english_channels_without_duplicates['text_preprocessed'].str.split().apply(len) > considered_short_tokens]
#write short_df to memory
write_df_in_chunks(short_df, output_path_preprocessed_messages_only_with_short_messages)
del short_df
gc.collect()

#SPAM DATAFRAM, naturally  it contains also short messagess
spam_df = (
    df_preprocessed_non_empty_english_channels
    .groupby(['channel_id', 'text_preprocessed'])
    .size()
    .reset_index(name='count_spam')
    .query('count_spam > @considered_spam_threshold')
    .sort_values(['channel_id', 'count_spam'], ascending=[True, False])
)
#write spam_df to file
spam_df.to_csv(output_path_preprocessed_messages_only_with_spam_messages, sep='\t', index=False, compression='gzip')
del spam_df
gc.collect()


#CHANNELS WITHOUT MESSAGES DATAFRAME
expected_channels = set(df_first_nodes['type_and_id'])  # all channels we expect to find from the input nodes file
seen_channels_any = set(df_preprocessed_non_empty_english_channels['channel_id'].unique())  # all channels actually present in the preprocessed messages
missing_channels_any = sorted(expected_channels - seen_channels_any)  # channels that are expected but have no messages
df_channels_without_messages = pd.DataFrame({'channel_id': missing_channels_any})  # dataframe with the list of channels without messages

#WRITE DATAFRAMES TO FILES
df_channels_without_messages.to_csv(output_path_channels_without_message, sep='\t', index=False, compression='gzip')
df_preprocessed_non_empty_english_channels.to_csv(output_path_preprocessed_english_messages, sep='\t', index=False, compression='gzip')

with open("preProcessing_completed_successfully.txt", "w") as f:
    f.write("preProcessing completata con successo.\n")
