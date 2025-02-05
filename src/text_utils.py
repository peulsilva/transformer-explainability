import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet, stopwords
from nltk import pos_tag

# Download necessary NLTK resources
import string
from collections import Counter
import re

# Download stopwords if not already present
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('punkt')

def remove_stopwords(text):
    stop_words = set(stopwords.words('english'))  # Load English stopwords
    stop_words.add('the')
    stop_words.add('a')
    words = word_tokenize(text)  # Tokenize the text
    filtered_text = [word for word in words if word.lower() not in stop_words]  # Remove stopwords
    return ' '.join(filtered_text)  # Join words back into a string


def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


def get_word_frequencies(text):
    # Remove punctuation and convert to lowercase
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count word occurrences
    word_counts = Counter(words)
    
    # Convert to vector format (list of tuples)
    word_vector = list(word_counts.items())
    
    return word_vector


def get_wordnet_pos(word):
    """Map POS tag to first character for WordNet lemmatizer"""

    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)  # Default to NOUN if tag is not found

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)  # Tokenize text
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]  # Lemmatize words
    return ' '.join(lemmatized_words)
