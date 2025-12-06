# ============================================
# preprocessor.py - Text Preprocessing Pipeline
# ============================================
# This file handles all the text preprocessing:
# 1. Tokenization (splitting text into words)
# 2. Stopword removal (removing common words like "the", "is", etc.)
# 3. Lemmatization (converting words to base form: "running" -> "run")

import re
import nltk
nltk.download('wordnet', quiet=True)  # download wordnet data for lemmatization
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()


# ============================================
# STOPWORDS LIST
# ============================================
# These are common words that don't carry much meaning.
# We remove them to focus on important words only.

STOPWORDS = {
    # Articles
    'a', 'an', 'the',
    
    # Pronouns
    'i', 'me', 'my', 'myself',
    'you', 'your', 'yours', 'yourself', 'yourselves',
    'he', 'him', 'his', 'himself',
    'she', 'her', 'hers', 'herself',
    'it', 'its', 'itself',
    'we', 'us', 'our', 'ours', 'ourselves',
    'they', 'them', 'their', 'theirs', 'themselves',
    'who', 'whom', 'whose',
    'what', 'which',
    'this', 'that', 'these', 'those',
    
    # Prepositions
    'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
    'from', 'up', 'down', 'out', 'off', 'over', 'under',
    'into', 'onto', 'upon', 'through', 'during', 'before',
    'after', 'above', 'below', 'between', 'among', 'about',
    'against', 'along', 'around', 'behind', 'beside', 'besides',
    'beyond', 'within', 'without',
    
    # Conjunctions
    'and', 'or', 'but', 'nor', 'so', 'yet',
    'if', 'then', 'else', 'when', 'where', 'why', 'how',
    'because', 'although', 'though', 'while', 'unless', 'until',
    'whether', 'as', 'than',
    
    # Auxiliary/Modal Verbs
    'is', 'am', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'having',
    'do', 'does', 'did', 'doing', 'done',
    'will', 'would', 'shall', 'should',
    'can', 'could', 'may', 'might', 'must',
    
    # Common Verbs
    'get', 'got', 'getting',
    'make', 'made', 'making',
    'go', 'goes', 'went', 'going', 'gone',
    'come', 'came', 'coming',
    'take', 'took', 'taking', 'taken',
    'see', 'saw', 'seen', 'seeing',
    'know', 'knew', 'known', 'knowing',
    'think', 'thought', 'thinking',
    'say', 'said', 'saying',
    'give', 'gave', 'given', 'giving',
    
    # Adverbs
    'not', 'no', 'yes',
    'very', 'really', 'just', 'only', 'also', 'too',
    'now', 'then', 'here', 'there', 'where', 'when',
    'always', 'never', 'often', 'sometimes', 'usually',
    'again', 'ever', 'already', 'still', 'even',
    'well', 'back', 'away', 'almost', 'enough',
    
    # Quantifiers
    'all', 'any', 'both', 'each', 'every',
    'few', 'many', 'much', 'more', 'most', 'less', 'least',
    'some', 'several', 'such', 'other', 'another',
    'none', 'either', 'neither',
    
    # Others
    'own', 'same', 'different',
    'first', 'last', 'next',
    'new', 'old', 'good', 'bad',
    'big', 'small', 'long', 'short',
    'however', 'therefore', 'thus', 'hence',
    'etc', 'eg', 'ie',
    
    # Contractions (leftover parts after tokenization)
    's', 't', 'd', 'll', 've', 're', 'm',
    'don', 'doesn', 'didn', 'won', 'wouldn', 'couldn', 'shouldn',
    'isn', 'aren', 'wasn', 'weren', 'hasn', 'haven', 'hadn'
}


# ============================================
# TOKENIZATION FUNCTION
# ============================================
# This splits text into individual words (tokens).
# It also converts everything to lowercase.
# We IGNORE pure numbers - they don't help in search.
# We KEEP alphanumeric words like "gpt4", "covid19" - they're meaningful.

def tokenize(text):
    """
    Splits text into tokens (words).
    - converts to lowercase
    - keeps words longer than 1 character
    - IGNORES pure numbers (e.g., "2024", "100") - not useful for search
    - KEEPS alphanumeric words (e.g., "gpt4", "bert") - they're meaningful
    """
    text = text.lower()  # convert to lowercase
    tokens = []
    current_word = ""
    
    # go through each character
    for char in text:
        if char.isalpha():
            # it's a letter, add to current word
            current_word += char
        elif char.isdigit():
            # it's a number, add to current word
            current_word += char
        else:
            # it's a space or punctuation - word is done
            if current_word:
                # only keep if length > 1 AND not a pure number
                if len(current_word) > 1 and not current_word.isdigit():
                    tokens.append(current_word)
                current_word = ""
    
    # don't forget the last word!
    if current_word:
        current_word = current_word.strip('-')
        # only keep if length > 1 AND not a pure number
        if len(current_word) > 1 and not current_word.isdigit():
            tokens.append(current_word)
    
    return tokens
# ============================================
# STOPWORD REMOVAL FUNCTION
# ============================================
# Removes common words that don't help with search.

def remove_stopwords(tokens):
    """
    Removes stopwords from the token list.
    Input: ['the', 'machine', 'learning', 'is', 'great']
    Output: ['machine', 'learning', 'great']
    """
    tokens_without_stopwords = []
    for token in tokens:
        if token not in STOPWORDS:
            tokens_without_stopwords.append(token)
    return tokens_without_stopwords



# ============================================
# LEMMATIZATION FUNCTION
# ============================================
# Converts words to their base form.
# Example: "running" -> "run", "studies" -> "study"
# We use NLTK's WordNet lemmatizer here.
# (You can also implement your own stemmer - see commented code below)

def lemmatize(word):
    """
    Converts a word to its base form (lemma).
    First tries as a verb, then as a noun.
    """
    # try lemmatizing as verb first ("running" -> "run")
    lemma = lemmatizer.lemmatize(word, pos='v')
    if lemma != word:
        return lemma
    # if no change, try as noun ("studies" -> "study")
    return lemmatizer.lemmatize(word, pos='n')


# ============================================
# MAIN PREPROCESSING FUNCTION
# ============================================
# This is what we call from main.py
# It takes documents and returns preprocessed tokens.

def preprocessing(dictionary):
    """
    Main preprocessing function.
    
    Input: {doc_id: "raw text...", doc_id: "raw text...", ...}
    Output: {doc_id: ["token1", "token2", ...], ...}
    
    Steps:
    1. Tokenize (split into words, lowercase)
    2. Remove stopwords
    3. Lemmatize (get base form of words)
    """
    processed_dict = {}
    
    for doc_id, text in dictionary.items():
        
        # handle edge case: if text is None or not a string
        if not isinstance(text, str):
            processed_dict[doc_id] = []
            continue
        
        # step 1: tokenize the text
        tokens = tokenize(text)
        
        # step 2: remove stopwords
        tokens = remove_stopwords(tokens)
        
        # step 3: lemmatize each token
        lemmatized_tokens = [lemmatize(token) for token in tokens]
        
        # save the processed tokens
        processed_dict[doc_id] = lemmatized_tokens
    
    return processed_dict