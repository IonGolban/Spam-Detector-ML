import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import data_man

path_bare = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\bare"
path_lemm = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\lemm"
path_lemm_stop = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\lemm_stop"
path_stop = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\stop"

path_bare_data = dict()
path_lemm_data = dict()
path_lemm_stop_data = dict()
path_stop_data = dict()


def procces_data(path):
    imported_data = data_man.import_data_and_labels(path_bare)

    tokenized_data = data_man.tokenize_data(imported_data)

    filtered_data = data_man.eliminate_stop_words(tokenized_data)
    return filtered_data

def create_attributes(data):
    attributes = set()



print(procces_data(path_bare))
