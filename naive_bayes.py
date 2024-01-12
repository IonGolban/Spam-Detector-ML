import copy
import nltk
import ssl
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from utils import data_man
import matplotlib.pyplot as plt

path_bare = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\bare_naiv\antrenare_bare"
path_lemm = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\lemm_naiv\antr"
path_lemm_stop = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\lem_stop_naiv\antr"
path_stop = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\stop_naiv\antr"
path_bare_testare = r"C:\Users\uig26544\PycharmProjects\Detector-Spam-ML\lingspam_public\stop_naiv\test"


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


def create_attributes(data):
    spam_not_spam = {}
    spam_not_spam['spam'] = {}
    spam_not_spam['non_spam'] = {}
    for key in data:
        if data_man.check_its_spam_msg(key):
            for word in data[key]:
                if word not in spam_not_spam['spam']:
                    spam_not_spam['spam'][word] = 1
                else:
                    spam_not_spam['spam'][word] += 1
        else:
            for word in data[key]:
                if word not in spam_not_spam['non_spam']:
                    spam_not_spam['non_spam'][word] = 1
                else:
                    spam_not_spam['non_spam'][word] += 1

    return spam_not_spam


def Naive_Bayes_Learn(data, mails, spam_mail, non_spam_mails):
    total_number_of_mails = mails
    priori_prob_of_spam = spam_mail / total_number_of_mails  # P(spam) priori
    priori_prob_of_non_spam = non_spam_mails / total_number_of_mails  # P(non_spam) priori

    unique_words = {}
    number_of_all_spam_words = 0
    number_of_all_non_spam_words = 0
    for key in data['spam']:
        number_of_all_spam_words += data['spam'][key]
        if key not in unique_words:
            unique_words[key] = 1
        else:
            unique_words[key] += 1
    for key in data['non_spam']:
        number_of_all_non_spam_words += data['non_spam'][key]
        if key not in unique_words:
            unique_words[key] = 1
        else:
            unique_words[key] += 1

    total_number_of_unique_words = len(unique_words)

    conditional_propabilities = {}
    conditional_propabilities[
        'spam'] = {}  # P(word|spam) = (frecventa_cuvantului + 1)/(frecventa_toate_cuvinte_spam + secventa_toate_cuvinte_dictionar)
    conditional_propabilities[
        'non_spam'] = {}  # P(word|non_spam) = (frecventa_cuvantului + 1)/(frecventa_toate_cuvinte_non_spam + secventa_toate_cuvinte_dictionar)
    epsilon = 1e-10
    for key in data['spam']:
        conditional_propabilities['spam'][key] = (data['spam'][key] + epsilon) / (
                number_of_all_spam_words + total_number_of_unique_words)

    for key in data['non_spam']:
        conditional_propabilities['non_spam'][key] = (data['non_spam'][key] + epsilon) / (
                number_of_all_non_spam_words + total_number_of_unique_words)

    return conditional_propabilities, priori_prob_of_spam, priori_prob_of_non_spam, number_of_all_spam_words, number_of_all_non_spam_words, total_number_of_unique_words


# calculam probabilitatile posterioare
# P(spam|email) = P(spam) * prod(P(word_i|spam))
# P(non_spam|email) = P(non_spam) * prod(P(word_i|non_spam))
def Clasify_New_Instance(data, cond, priori_spam, priori_non_spam, spm, nnspm, total):
    right_predictions = 0
    total_number_of_instances = 0
    for key in data:
        total_number_of_instances += 1
        posteriori_prob_for_spam = priori_spam
        posteriori_prob_for_non_spam = priori_non_spam
        prod_spam = 1
        prod_non_spam = 1

        epsilon = 1e-10
        for word in data[key]:
            if word in cond['spam']:
                prod_spam *= cond['spam'][word]
            elif word not in cond['spam']:
                # Aplicam Laplace
                prod_spam *= (epsilon / (spm + total))

            if word in cond['non_spam']:
                prod_non_spam *= cond['non_spam'][word]
            elif word not in cond['non_spam']:
                # Aplicam Laplace
                prod_non_spam *= (epsilon / (nnspm + total))

        posteriori_prob_for_non_spam *= prod_non_spam
        posteriori_prob_for_spam *= prod_spam

        if posteriori_prob_for_non_spam < posteriori_prob_for_spam:
            # print(f"Emailul cu numele {key} va fi clasificat ca email spam!!")
            if 'spmsga' in key:
                right_predictions += 1
        else:
            # print(f"Emailul cu numele {key} va fi clasificat ca email non_spam!!")
            if 'spmsga' not in key:
                right_predictions += 1

    return right_predictions / total_number_of_instances


def Leave_One_Out_Validation(proc_data, total_mails, total_spam_mails, total_non_spam_mails):
    correct_predictions = 0
    for instance in proc_data.keys():

        proc_data_learn = copy.deepcopy(proc_data)
        proc_data_learn.pop(instance)
        total_non_spam_mails_learn = total_non_spam_mails
        total_spam_mails_learn = total_spam_mails
        total_mails_learn = total_mails

        total_mails_learn -= 1
        if data_man.check_its_spam_msg(instance):
            total_spam_mails_learn -= 1
        else:
            total_non_spam_mails_learn -= 1
        attributes = create_attributes(proc_data_learn)

        cond, priori_spam, priori_non_spam, spm, nnspm, total = Naive_Bayes_Learn(attributes, total_mails_learn,
                                                                                  total_spam_mails_learn,
                                                                                  total_non_spam_mails_learn)
        correct_predictions += Clasify_New_Instance({instance: proc_data[instance]}, cond, priori_spam, priori_non_spam,
                                                    spm, nnspm,
                                                    total)

    return correct_predictions / total_mails


proc_data, total_mails, total_spam_mails, total_non_spam_mails = procces_data(path_stop)
atributes = create_attributes(proc_data)
cond, priori_spam, priori_non_spam, spm, nnspm, total = Naive_Bayes_Learn(atributes, total_mails, total_spam_mails,
                                                                          total_non_spam_mails)

proc_data_test, _, _, _ = procces_data(path_bare_testare)

accuracy = Clasify_New_Instance(proc_data_test, cond, priori_spam, priori_non_spam, spm, nnspm, total)
print(f'-------- Acuratetea programului este {round(accuracy * 100, 2)}% --------')

accuracy_leave_one_out = Leave_One_Out_Validation(proc_data, total_mails, total_spam_mails, total_non_spam_mails)
print(f'-------- Acuratetea programului este la leave one out {round(accuracy_leave_one_out * 100, 2)}% --------')

trivial = 0.5
algoritmi = ['N.B', 'ID3', 'ADABOOST', 'Strategie Trivială']
acurateti = [accuracy * 100, 0.9586 * 100, 0.9483 * 100, trivial * 100]

plt.bar(algoritmi, acurateti)

plt.xlabel('Algoritmi')
plt.ylabel('Acuratețe (%)')
plt.title('Performanța Algoritmilor de Clasificare set: Stop')

plt.show()
