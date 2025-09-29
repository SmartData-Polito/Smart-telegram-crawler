# Fare preprocessing dei testi:
import os
import re
from typing import Callable, Union

# import spacy
# from sklearn.feature_extraction.text import TfidfVectorizer
# from tqdm import tqdm
from unidecode import unidecode
import langdetect
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os
import pandas as pd
from glob import glob


class PreProcessing:
    """Class for performing text preprocessing operations.

    Args:
        noadverbs (bool, optional): Flag to remove adverbs from the text. Defaults to False.
        noadjectives (bool, optional): Flag to remove adjectives from the text. Defaults to False.
        noverbs (bool, optional): Flag to remove verbs from the text. Defaults to False.
        noentities (bool, optional): Flag to remove named entities from the text. Defaults to False.
        language (str, optional): Language for the Spacy model. Defaults to 'en'.
        remove_list (bool, optional): Flag to remove a list of words from the text. Defaults to False.

    Attributes:
        noadverbs (bool): Flag to remove adverbs from the text.
        noadjectives (bool): Flag to remove adjectives from the text.
        noverbs (bool): Flag to remove verbs from the text.
        noentities (bool): Flag to remove named entities from the text.
        language (str): Language for the Spacy model.
        remove_list (bool): Flag to remove a list of words from the text.
        punctuation (str): Regular expression pattern for removing punctuation.
        nlp (spacy.Language): Spacy language model.
        stopwords (list): List of stopwords.

    Methods:
        lowercase_unidecode: Converts text to lowercase and removes diacritics.
        remove_urls: Removes URLs from the text.
        remove_tweet_marking: Removes Twitter mentions and hashtags from the text.
        remove_punctuation: Removes punctuation from the text.
        remove_repetion: Removes repeated words from the text.
        append_stopwords_list: Appends additional stopwords to the existing list.
        remove_stopwords: Removes stopwords from the text.
        remove_n: Removes words with length less than or equal to n from the text.
        remove_numbers: Removes or filters out numbers from the text.
        remove_gerund: Removes gerund endings from verbs in the text.
        remove_infinitive: Removes infinitive endings from verbs in the text.
        filter_by_idf: Filters out words based on their inverse document frequency.

    """

    def __init__(self, noadverbs: bool = False, noadjectives: bool = False, noverbs: bool = False,
                 noentities: bool = False, language: str = 'en', remove_list: bool = False,stopwords=[]):
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
                r'|!|"|#|\$|%|&|\'|\(||!|"|#|\$|%|&|\'|\(||\*|\+|,|-|\.|\/|'
                r':|;|<|=|>|\?|\@||||\^|_|`|\{|\}|~|\||'
                r'\r\n|\n|\r|\\\)'
        )
        # self.nlp = self._load_spacy_model(language)
        # self.stopwords = [unidecode(x).lower() for x in list(self.nlp.Defaults.stop_words)]
        self.stopwords=stopwords



    
    def _process_text(self, text: Union[str, list], function: Callable) -> Union[str, list]:

        if isinstance(text, str):
            return function(text)
        elif isinstance(text, list):
            return [function(x) for x in text]
        return ''
    
    
    def lowercase_unidecode(self, text: Union[str, list]) -> Union[str, list]:
        """Convert the given text to lowercase and remove any diacritical marks (accents).

        Args:
            text (Union[str, list]): The text to be processed. It can be either a string or a list of strings.

        Returns:
            Union[str, list]: The processed text. If the input is a string, the output will be a string. If the input is a list,
            the output will be a list of strings.

        Example:
            >>> pre_processor = PreProcessor()
            >>> text = "Café"
            >>> pre_processor.lowercase_unidecode(text)
            'cafe'
        """
        from unidecode import unidecode
        text = self._process_text(text, lambda value: value.lower())
        text = self._process_text(text, unidecode)
        return text

    def remove_urls(self, text: Union[str, list]) -> Union[str, list]:
        """Removes URLs from the given text or list of texts.

        Args:
            text (Union[str, list]): The text or list of texts from which to remove URLs.

        Returns:
            Union[str, list]: The text or list of texts with URLs removed.

        """
        return self._process_text(text, lambda value: re.sub(r'http\S+ *', '', value).strip())

    def remove_tweet_marking(self, text: Union[str, list]) -> Union[str, list]:
        """Removes tweet markings (e.g., @mentions and #hashtags) from the given text.

        Args:
            text (Union[str, list]): The text or list of texts to process.

        Returns:
            Union[str, list]: The processed text or list of processed texts with tweet markings removed.
        """
        return self._process_text(text, lambda value: re.sub(r'(@|#)\S+ *', '', value).strip())

    def remove_html_tags(self, text: Union[str, list]) -> Union[str, list]:
        """Removes HTML tags from the given text.

        Args:
            text (Union[str, list]): The text or list of texts to process.

        Returns:
            Union[str, list]: The processed text or list of processed texts with HTML tags removed.
        """
        return self._process_text(text, lambda value: re.sub(r'<.*?> *', '', value).strip())

    def remove_punctuation(self, text: Union[str, list]) -> Union[str, list]:
        """Removes punctuation from the given text.

        Args:
            text (Union[str, list]): The text from which punctuation needs to be removed.

        Returns:
            Union[str, list]: The text with punctuation removed.
        """
        text = self._process_text(text, lambda value: re.sub(self.punctuation, ' ', value))
        text = self._process_text(text, lambda value: re.sub(' {2,}', ' ', value).strip())
        return text

    def remove_repetition(self, text: Union[str, list]) -> Union[str, list]:
        """Removes repeated words in the given text.

        Args:
            text (Union[str, list]): The input text or list of words.

        Returns:
            Union[str, list]: The processed text with repeated words removed.

        """
        return self._process_text(text, lambda value: re.sub(r'\b(\w+)\s+\1\b', r'\1', value))

    def append_stopwords_list(self, stopwords: list) -> None:
        """Appends additional stopwords to the existing list of stopwords.

        Parameters:
        stopwords (list): A list of stopwords to be appended.

        """
        self.stopwords.extend(stopwords)

    def remove_stopwords(self, text: Union[str, list]) -> Union[str, list]:
        """Removes stopwords from the given text.

        Args:
            text (Union[str, list]): The input text from which stopwords need to be removed.

        Returns:
            Union[str, list]: The processed text with stopwords removed.

        """
        return self._process_text(text, lambda value: re.sub(rf'\b({"|".join(self.stopwords)})\b *', '', value).strip())

    

    def remove_n(self, text: Union[str, list], n: int) -> Union[str, list]:
        """Removes words of length 1 to n followed by the word 'pri' from the given text.

        Args:
            text (Union[str, list]): The input text or list of texts to process.
            n (int): The maximum length of words to remove.

        Returns:
            Union[str, list]: The processed text or list of processed texts.

        """
        return self._process_text(text, lambda value: re.sub(rf'(\b|^)\w{{1,{n}}}(\b|$) ?', '', value).strip())

    def remove_numbers(self, text: Union[str, list], mode: str = 'replace') -> Union[str, list]:
        """Removes or replaces numbers in the given text.

        Args:
            text (Union[str, list]): The input text or list of texts.
            mode (str, optional): The mode of operation. Defaults to 'replace'.
                - 'filter': Removes the numbers from the text.
                - 'replace': Replaces the numbers with an empty string.

        Returns:
            Union[str, list]: The processed text or list of processed texts.
        """
        if mode == "filter":
            return self._process_text(text, lambda value: '' if re.search('[0-9]', value) else value)
        elif mode == "replace":
            return self._process_text(text, lambda value: re.sub('[0-9] *', '', value))

    def remove_gerund(self, text: Union[str, list]) -> Union[str, list]:
        """Removes the gerund form '-ndo' from the given text.

        Args:
            text (Union[str, list]): The input text or list of texts to process.

        Returns:
            Union[str, list]: The processed text with the gerund form removed.

        """
        return self._process_text(text, lambda value: re.sub(r'ndo\b', '', value))

    def remove_infinitive(self, text: Union[str, list]) -> Union[str, list]:
        """Removes the infinitive form of verbs from the given text.

        Args:
            text (Union[str, list]): The input text or list of texts to process.

        Returns:
            Union[str, list]: The processed text with infinitive forms removed.

        """
        return self._process_text(text, lambda value: re.sub(r'r\b', '', value))
    
    
    def detect_language(self,text):
        import langdetect
        try:
            d=langdetect.detect_langs(text)
            # Trasforma la lista in un dizionario
            langs_dict = {lang.lang: lang.prob for lang in d}
            best_lang=max(langs_dict,key=langs_dict.get)
            best_lang=best_lang if langs_dict[best_lang]>=0.7 else 'unk'
            return best_lang    
        except langdetect.LangDetectException as e:
            return 'unk'
        return None

from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)

# here the funziona to call to preprocess the text
def preprocess_text(text,stopwords=stopwords):
    try: 
        pp=PreProcessing(language='en',stopwords=stopwords)

        # Preprocessing pipeline
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
        result = (text_clean, lang)
    except Exception:
        return ("", "unk")
    if not (isinstance(result, tuple) and len(result) == 2):
        print("----\n-----\n error in preprocess_text ------\n------\n")
        result = ("", "unk")
    return result



    
#-------------------
#START



os.environ["TOKENIZERS_PARALLELISM"] = "false"


# Path to the final preprocessed file
output_path_preprocessed_messages = "../material/preprocessed_messages.tsv.gz"
output_path_preprocessed_english_messages = "../material/preprocessed_english_messages.tsv.gz"
output_path_preprocessed_messages_only_with_short_messages = "../material/preprocessed_short_messages.tsv.gz"
output_path_preprocessed_messages_only_with_spam_messages = "../material/preprocessed_spam_messages.tsv.gz"
output_path_channels_without_message = "../material/channels_without_message.tsv.gz"
input_path_df_first_nodes = "../material/first_nodes.csv.gz"
extracted_dir = '../../../telegram_2024/usc-tg-24-us-election/extracted'

df_first_nodes = pd.read_csv(input_path_df_first_nodes)

print(df_first_nodes.head())

channels_without_message = []

# If the file exists, load it directly and skip the rest
if os.path.exists(output_path_preprocessed_messages):
    print("--- File already exists: {}".format(output_path_preprocessed_messages))
    df_preprocessed_non_empty_channels = pd.read_csv(output_path_preprocessed_messages, sep='\t', compression='gzip')
    print("--- File loaded with {} preprocessed messages.".format(len(df_preprocessed_non_empty_channels)))
else:
    print("--- File not found, proceeding with preprocessing...")

    def process_file(args):
        file, channel_id, token = args
        try:
            df = pd.read_csv(file, sep='\t', compression='gzip', usecols=['text', 'timestamp'])
            df = df.dropna(subset=['text'])
            df['text'] = df['text'].astype(str)
            pairs = df['text'].apply(preprocess_text)
            
            #count the touple of length 2 (valids) and the void ones (invalids)
            valid   = sum(1 for p in pairs if isinstance(p, tuple) and len(p)==2)
            invalid = len(pairs) - valid
            #decomment to debug
            #print(f"--- valid pairs: {valid}, invalid pairs: {invalid}")
            #print("pairs", pairs)
            df['text_preprocessed'] = [p[0] for p in pairs]
            df['language']          = [p[1] for p in pairs]

            # 4) Filter and return immediatly
            df = df[df['text_preprocessed'] != ""]
            if df.empty:
                return None
                 
            
            df['channel_id'] = channel_id
            df['token'] = token
            return df if not df.empty else None
        except Exception as e:
            print(f"--- Error in file {file}: {type(e).__name__}: {e}")
            return None

    # Compute file_args and channels_without_message
    count_first_nodes = 0
    count_channels_without_message = 0
    file_args = []
    for _, row in df_first_nodes.iterrows():
        count_first_nodes += 1
        channel_id = row['type_and_id']
        token = row['token']
        channel_path = os.path.join(extracted_dir, channel_id)
        files = glob(os.path.join(channel_path, '[0-9][0-9][0-9][0-9]-[0-1][0-9].tsv.gz'))

        if not os.path.isdir(channel_path) or not files:
            channels_without_message.append(channel_id)
            count_channels_without_message += 1
            print(f"→ No data for {channel_id} | Total missing: {count_channels_without_message}")
            continue

        print(f"\n\n Found files for {channel_id}:\n{files}\n")

        file_args.extend([(file, channel_id, token) for file in files])

    print("--- Number of messages in file_args:", str(len(file_args)))
    print("--- Channels without messages count:", count_channels_without_message)
    print("--- First nodes count:", count_first_nodes)
    print("--- Number of distinct channel_ids in file_args:(subtraction of the above two)", len({entry[1] for entry in file_args}))

    # Multiprocessing
    results = []
    with Pool(cpu_count()) as pool:
        pbar = tqdm(total=len(file_args))
        for res in pool.imap_unordered(process_file, file_args):
            pbar.update(1)
            results.append(res)

    all_english_messages = [df for df in results if df is not None]
    df_preprocessed_non_empty_channels = pd.concat(all_english_messages, ignore_index=True)


if os.path.exists(output_path_channels_without_message):
    print("--- File already exists: {}".format(output_path_channels_without_message))
    df_channels_without_message = pd.read_csv(output_path_channels_without_message, sep='\t', compression='gzip')
    print("--- File channels_without_messages loaded with length = {}".format(len(df_channels_without_message)))
else:
    df_channels_without_message = pd.DataFrame({'channel_id': channels_without_message})
    
    
# Clean-up and filtering
df_channels_without_message = df_channels_without_message.dropna(subset=['channel_id'])
df_channels_without_message = df_channels_without_message[
    ~df_channels_without_message['channel_id'].isin(df_preprocessed_non_empty_channels['channel_id'])]
df_channels_without_message.to_csv(output_path_channels_without_message, sep='\t', index=False, compression='gzip')

#Clean-up and filtering
#--------
#create dataframe that count spam messages
df_preprocessed_non_empty_channels_spam_messages = (
    df_preprocessed_non_empty_channels
    .groupby(['channel_id', 'text_preprocessed'])
    .size()
    .reset_index(name='count')
    .query('count > 10')   
    .sort_values(['channel_id', 'count'], ascending=[True, False])
)
print("---len dataframe with spam messages: ", len(df_preprocessed_non_empty_channels_spam_messages))
df_preprocessed_non_empty_channels_spam_messages.to_csv(output_path_preprocessed_messages_only_with_spam_messages, sep='\t', index=False, compression='gzip')

#clean up and filtering and dividing channels with short and long messages
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels.dropna(subset=['channel_id', 'text_preprocessed'])
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels.astype(str)
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels[
    df_preprocessed_non_empty_channels['text_preprocessed'].apply(lambda x: isinstance(x, str))]
df_preprocessed_non_empty_channels['date'] = pd.to_datetime(df_preprocessed_non_empty_channels['timestamp'], unit='s')
print("len before drop_duplicates:", len(df_preprocessed_non_empty_channels))
df_preprocessed_non_empty_channels.drop_duplicates(subset=['text_preprocessed'], inplace=True)
print("len before dropping short message:", len(df_preprocessed_non_empty_channels))
df_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['text_preprocessed'].apply(len) > 20]
df_preprocessed_non_empty_channels_only_with_short_messages = df_preprocessed_non_empty_channels[df_preprocessed_non_empty_channels['text_preprocessed'].apply(len) <= 20]
df_preprocessed_non_empty_channels.to_csv(output_path_preprocessed_messages, sep='\t', index=False, compression='gzip')
df_preprocessed_non_empty_channels_only_with_short_messages.to_csv(output_path_preprocessed_messages_only_with_short_messages, sep='\t', index=False, compression='gzip')

# Filter English-only messages
df_english_preprocessed_non_empty_channels = df_preprocessed_non_empty_channels.copy()
df_english_preprocessed_non_empty_channels = df_english_preprocessed_non_empty_channels[df_english_preprocessed_non_empty_channels['language'] == 'en']
df_english_preprocessed_non_empty_channels.to_csv(output_path_preprocessed_english_messages, sep='\t', index=False, compression='gzip')

