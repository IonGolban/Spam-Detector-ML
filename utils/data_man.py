import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

imported_data = dict()


def import_data_and_labels(path):
    data = dict()
    for folder in os.listdir(path):
        for file in os.listdir(os.path.join(path, folder)):
            with open(os.path.join(os.path.join(path, folder), file), 'r') as f:
                data[file] = f.read()
    return data


def tokenize_data(imported_data):
    return {key: word_tokenize(imported_data[key]) for key in imported_data}


def eliminate_stop_words(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = dict()
    for key in tokens:
        filtered_tokens[key] = [word for word in tokens[key] if word.lower() not in stop_words and word.isalpha()]
    return filtered_tokens


def stemmer_data(tokens):
    stemmer = nltk.PorterStemmer()
    stemmed_tokens = dict()
    for key in tokens:
        stemmed_tokens[key] = [stemmer.stem(word) for word in tokens[key]]
    return stemmed_tokens


def check_its_spam_msg(name_of_file):
    return 'spmsga' in name_of_file