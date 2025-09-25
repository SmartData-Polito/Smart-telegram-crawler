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
        self.punctuation = (
                r'\(|!|"|#|\$|%|&|\'|\(|\)|\*|\+|,|-|\.|\/|'
                r':|;|<|=|>|\?|\@||||\^|_|`|\{|\}|~|\||'
                r'\r\n|\n|\r|\\\)'
        )
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

#process_file function definition
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
#input path
extracted_dir = '../../../telegram_2024/usc-tg-24-us-election/extracted'
level_depth = args.input
input_path_df_political_nodes = f'../results/levels/level_{level_depth}/preProcessing/nodes_level_{level_depth}.csv.gz'
# Output paths
output_path_preprocessed_messages = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_messages_level_{level_depth}.tsv.gz"
output_path_preprocessed_english_messages = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_english_messages_level_{level_depth}.tsv.gz"
output_path_preprocessed_messages_only_with_short_messages = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_short_messages_level_{level_depth}.tsv.gz"
output_path_preprocessed_messages_only_with_spam_messages = f"../results/levels/level_{level_depth}/preProcessing/preprocessed_spam_messages_level_{level_depth}.tsv.gz"
output_path_channels_without_message = f"../results/levels/level_{level_depth}/preProcessing/channels_without_message_level_{level_depth}.tsv.gz"

#create directorie
level_dir = f"../results/levels/level_{level_depth}/preProcessing/"
os.makedirs(level_dir, exist_ok=True)

# Read input list
channels_without_message = []
df_first_nodes = pd.read_csv(input_path_df_political_nodes)
print(df_first_nodes.head())

if os.path.exists(output_path_preprocessed_messages):
    print("--- File already exists: {}".format(output_path_preprocessed_messages))
    df_preprocessed_non_empty_channels = pd.read_csv(output_path_preprocessed_messages, sep='\t', compression='gzip')
    print("--- File loaded with {} preprocessed messages.".format(len(df_preprocessed_non_empty_channels)))
else:
    file_args = []
    for _, row in df_first_nodes.iterrows():
        channel_id = row['type_and_id']
        channel_path = os.path.join(extracted_dir, channel_id)
        files = glob(os.path.join(channel_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz'))
        if not os.path.isdir(channel_path) or not files:
            channels_without_message.append(channel_id)
            continue
        file_args.extend([(file, channel_id) for file in files])

    #multiprocessing
    dfs = []
    with Pool(cpu_count()) as pool:
        for res in tqdm(pool.imap_unordered(process_file, file_args), total=len(file_args)): #file_args = list of (file, channel_id)
            if res is not None:
                dfs.append(res)
    # Concatenate directly
    if dfs:
        df_preprocessed_non_empty_channels = pd.concat(dfs, ignore_index=True)
    else:
        df_preprocessed_non_empty_channels = pd.DataFrame()

if not os.path.exists(output_path_channels_without_message):
    df_channels_without_message = pd.DataFrame({'channel_id': channels_without_message})
    df_channels_without_message = df_channels_without_message.dropna(subset=['channel_id'])
    df_channels_without_message = df_channels_without_message[
        ~df_channels_without_message['channel_id'].isin(df_preprocessed_non_empty_channels['channel_id'])]
    df_channels_without_message.to_csv(output_path_channels_without_message, sep='\t', index=False, compression='gzip')

# Spam detection
spam_df = (
    df_preprocessed_non_empty_channels
    .groupby(['channel_id', 'text_preprocessed'])
    .size()
    .reset_index(name='count')
    .query('count > 6')
    .sort_values(['channel_id', 'count'], ascending=[True, False])
)
print("---len dataframe with spam messages: ", len(spam_df))
spam_df.to_csv(output_path_preprocessed_messages_only_with_spam_messages, sep='\t', index=False, compression='gzip')


# Short/Long message split
# Clean duplicates and short messages
print("len before drop_duplicates:", len(df_preprocessed_non_empty_channels))
df_preprocessed_non_empty_channels.drop_duplicates(subset=['text_preprocessed'], inplace=True)
print("len after drop_duplicates:", len(df_preprocessed_non_empty_channels))

short_df = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['text_preprocessed'].apply(len) <= 20]
print("len before dropping short message:", len(df_preprocessed_non_empty_channels))
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['text_preprocessed'].apply(len) > 20]
print("len after dropping short message:", len(df_preprocessed_non_empty_channels))

#writing in chunks the non empty channels and the one only with short messages
write_df_in_chunks(df_preprocessed_non_empty_channels, output_path_preprocessed_messages)
write_df_in_chunks(short_df, output_path_preprocessed_messages_only_with_short_messages)

# English filter
df_english = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['language'] == 'en']
df_english.to_csv(output_path_preprocessed_english_messages, sep='\t', index=False, compression='gzip')

with open("completed_successfully.txt", "w") as f:
    
    f.write("preProcessing completata con successo.\n")