import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import data_man

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def procces_data(path):
    imported_data = data_man.import_data_and_labels(path)

    tokenized_data = data_man.tokenize_data(imported_data)

    filtered_data = data_man.eliminate_stop_words(tokenized_data)
    total_mails = 0
    total_spam = 0
    total_non_spam = 0
    for key in filtered_data:
        total_mails += 1
        if data_man.check_its_spam_msg(key):
            total_spam += 1
        else:
            total_non_spam += 1

    return filtered_data, total_mails, total_spam, total_non_spam


def proc_data_for_adaboost(data):
    adaboost_data = {}
    adaboost_data['text'] = []
    adaboost_data['label'] = []

    for file_name in data:
        email_content = data[file_name]
        email_content = str(' '.join(email_content))
        adaboost_data['text'].append(email_content)
        if data_man.check_its_spam_msg(file_name):
            adaboost_data['label'].append('spam')
        else:
            adaboost_data['label'].append('non_spam')

    return adaboost_data

path_lemm_stop = "/Users/sebastiandluman/Desktop/ML_Project/Spam-Detector-ML/lingspam_public/lemm_stop"
proc_data, _, _, _ = procces_data(path_lemm_stop)
adaboost_data = proc_data_for_adaboost(proc_data)

df = pd.DataFrame(adaboost_data)

vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(df['text'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.1, random_state=42)

adaboost_classifier = AdaBoostClassifier(n_estimators=50, random_state=42)
adaboost_classifier.fit(X_train, y_train)
predictions = adaboost_classifier.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
print(f'Acuratețe Adaboost: {accuracy * 100:.2f}%')