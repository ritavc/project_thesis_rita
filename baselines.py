import itertools
import pandas as pd
import ast
import numpy as np
import nltk, re, string, collections
import operator
from nltk.util import ngrams

df_cat = pd.read_csv(
    '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')  # , converters={'parentid': lambda x: str(x)})
df_cat = df_cat.fillna(-1)
df_cat = df_cat.astype(int)


def empty_initial_categories_dict():
    list_cats_initial = []
    tree_dict = dict()
    root_cats = df_cat.loc[(df_cat.parentid == -1)]
    root_cats = list(root_cats.categoryid)
    root_cats.sort()
    for cat in root_cats:
        tree_dict[cat] = []
    # print(root_cats)
    return tree_dict


def category_tree():
    tree_dict_l1_original = empty_initial_categories_dict()  ###level 1
    list_root_categories = list(tree_dict_l1_original.keys())
    aux_dict_2 = {}
    for cat in list_root_categories:  ###level 2
        categories_next_level = df_cat.loc[(df_cat.parentid == int(cat))]
        l = list(categories_next_level.categoryid)
        l.sort()
        tree_dict_l1_original[cat] = l
        for elem in l:
            aux_dict_2[elem] = cat
    return aux_dict_2, tree_dict_l1_original


general_tree_category = category_tree()
hierarchy_aux_dict = general_tree_category[0]

unique_cats_level2 = list(itertools.chain.from_iterable(general_tree_category[1].values()))
unique_cats_level2.sort()
# print(unique_cats_level2)
# print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))
df_train = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_train_set.csv')
df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_val_set.csv')
df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_test_set.csv')
df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                    df_train['sequence_cats_level2']]
df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                         df_validation['sequence_cats_level2']]
df_test['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                   df_test['sequence_cats_level2']]

total_categories_level2 = len(unique_cats_level2)


def most_common():
    df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')
    k = 5
    top_k_common = df['level2'].value_counts()[:5].index.tolist()
    print("most common")
    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        real_cat = row['next_cat_level2']
        predicted_cat = top_k_common[0]
        if predicted_cat == real_cat:
            cases_right_test += 1

    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))
    #print("recall_at_5 = %s" % (gotten_right_top_5 / nr_cases_test))
    #print("mrr_at_5 = %s" % (reciprocal_ranks / nr_cases_test))

most_common()

def assign_previous():
    print("assign previous")
    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        real_cat = row['next_cat_level2']
        last_category = row['sequence_cats_level2'][-1]
        predicted_cat = last_category
        if predicted_cat == real_cat:
            cases_right_test += 1

    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))
    #print("recall_at_5 = %s" % (gotten_right_top_5 / nr_cases_test))
    #print("mrr_at_5 = %s" % (reciprocal_ranks / nr_cases_test))

assign_previous()
def assign_previous_items_experience():
    print("assign previous ITEMS IDs experience")
    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        real_item = row['next_itemId']
        last_item = row['sequence_items'][-1]
        predicted_itemId = last_item
        if predicted_itemId == real_item:
            cases_right_test += 1

    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))
    #print("recall_at_5 = %s" % (gotten_right_top_5 / nr_cases_test))
    #print("mrr_at_5 = %s" % (reciprocal_ranks / nr_cases_test))
assign_previous_items_experience()

def bigrams():
    sequences_text = []
    grams_total = []

    for index, row in df_train.iterrows():
        sequence = row['sequence_cats_level2'] + [row['next_cat_level2']]
        sequences_text.extend(sequence)
        grams = ngrams(sequence, 2)
        grams_total += grams
    grams_freq = collections.Counter(grams_total)
    dict_start_cats_counts = {}
    for key, value in grams_freq.items():
        if key[0] in dict_start_cats_counts:
            dict_start_cats_counts[key[0]] = dict_start_cats_counts.get(key[0]) + value
        else:
            dict_start_cats_counts[key[0]] = value
    dict_transitions_smooth = {}
    for key, value in grams_freq.items():
        if key[0] in dict_transitions_smooth:
            dict_transitions_smooth[key[0]].update(
                {key[1]: (value + 1) / (dict_start_cats_counts.get(key[0]) + total_categories_level2)})
        else:
            dict_transitions_smooth[key[0]] = {
                key[1]: (value + 1) / (dict_start_cats_counts.get(key[0]) + total_categories_level2)}
    # print(dict_transitions_smooth)

    print("IN TRAIN SET:")
    # in train dataset:
    cases_right_train = 0
    nr_cases_train = 0
    for index, row in df_train.iterrows():
        nr_cases_train += 1
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat_train = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        search_dict = dict_transitions_smooth.get(last_category)
        max_prob = 0
        predicted_cat_train = -1
        for key, val in search_dict.items():
            if val > max_prob:
                max_prob = val
                predicted_cat_train = key
        real_cat_train = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat_train, real_cat_train))
        if predicted_cat_train == real_cat_train:
            cases_right_train += 1
    accuracy_train = cases_right_train / nr_cases_train
    print("accuracy train = {}; nr cases gotten right in train set = {}; total nr of cases in train set = {}".format(
        accuracy_train, cases_right_train, nr_cases_train))

    print("IN VALIDATION SET:")
    # in validation dataset:
    cases_right_val = 0
    nr_cases_val = 0
    for index, row in df_validation.iterrows():
        nr_cases_val += 1
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if last_category in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get(last_category)
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_val += 1
    accuracy_validation = cases_right_val / nr_cases_val
    print("accuracy validation = {}; nr cases gotten right in val set = {}; total nr of cases in val set = {}".format(
        accuracy_validation, cases_right_val, nr_cases_val))

    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    gotten_right_top_5 = 0
    reciprocal_ranks = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if last_category in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get(last_category)
            max_prob = 0
            predicted_cat = -1
            '''for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key'''
            new = dict(sorted(search_dict.items(), key=operator.itemgetter(1),
                              reverse=True))  # {k: v for k, v in sorted(search_dict.items(), key=lambda item: item[1])}
            predicted_cat = next(iter(new))
            top_5 = list(new.keys())[:5]
            if predicted_cat in top_5:
                gotten_right_top_5 += 1
                reciprocal_rank = 1 / (top_5.index(predicted_cat) + 1)
            reciprocal_ranks += reciprocal_rank
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_test += 1
    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))
    print("recall_at_5 = %s" % (gotten_right_top_5 / nr_cases_test))
    print("mrr_at_5 = %s" % (reciprocal_ranks / nr_cases_test))

print("****** 2-GRAMS *******")
bigrams()

def trigrams():
    sequences_text = []
    grams_total = []
    for index, row in df_train.iterrows():
        sequence = row['sequence_cats_level2'] + [row['next_cat_level2']]
        sequences_text.extend(sequence)
        grams = ngrams(sequence, 3)
        grams_total += grams
    grams_freq = collections.Counter(grams_total)
    dict_start_cats_counts = {}
    for key, value in grams_freq.items():
        if (key[0], key[1]) in dict_start_cats_counts:
            dict_start_cats_counts[(key[0], key[1])] = dict_start_cats_counts.get((key[0], key[1])) + value
        else:
            dict_start_cats_counts[(key[0], key[1])] = value
    dict_transitions_smooth = {}
    for key, value in grams_freq.items():
        if (key[0], key[1]) in dict_transitions_smooth:
            dict_transitions_smooth[(key[0], key[1])].update(
                {key[2]: (value + 1) / (dict_start_cats_counts.get((key[0], key[1])) + total_categories_level2)})
        else:
            dict_transitions_smooth[(key[0], key[1])] = {
                key[2]: (int(value) + 1) / (dict_start_cats_counts.get((key[0], key[1])) + total_categories_level2)}
    # print(dict_transitions_smooth)

    print("IN TRAIN SET:")
    # in train dataset:
    cases_right_train = 0
    nr_cases_train = 0
    for index, row in df_train.iterrows():
        nr_cases_train += 1
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        search_dict = dict_transitions_smooth.get((second_last_category, last_category))
        max_prob = 0
        predicted_cat_train = -1
        for key, val in search_dict.items():
            if val > max_prob:
                max_prob = val
                predicted_cat_train = key
        real_cat_train = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat_train, real_cat_train))
        if predicted_cat_train == real_cat_train:
            cases_right_train += 1
    accuracy_train = cases_right_train / nr_cases_train
    print("accuracy train = {}; nr cases gotten right in train set = {}; total nr of cases in train set = {}".format(
        accuracy_train, cases_right_train, nr_cases_train))

    print("IN VALIDATION SET:")
    # in validation dataset:
    cases_right_val = 0
    nr_cases_val = 0
    for index, row in df_validation.iterrows():
        nr_cases_val += 1
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if (second_last_category, last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get((second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_val += 1
    accuracy_validation = cases_right_val / nr_cases_val
    print("accuracy validation = {}; nr cases gotten right in val set = {}; total nr of cases in val set = {}".format(
        accuracy_validation, cases_right_val, nr_cases_val))

    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if (second_last_category, last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get((second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_test += 1
    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))


print("****** 3-GRAMS *******")


# trigrams()

def fourgrams():
    sequences_text = []
    grams_total = []
    for index, row in df_train.iterrows():
        sequence = row['sequence_cats_level2'] + [row['next_cat_level2']]
        sequences_text.extend(sequence)
        grams = ngrams(sequence, 4)
        grams_total += grams
    grams_freq = collections.Counter(grams_total)
    dict_start_cats_counts = {}
    for key, value in grams_freq.items():
        if (key[0], key[1], key[2]) in dict_start_cats_counts:
            dict_start_cats_counts[(key[0], key[1], key[2])] = dict_start_cats_counts.get(
                (key[0], key[1], key[2])) + value
        else:
            dict_start_cats_counts[(key[0], key[1], key[2])] = value
    dict_transitions_smooth = {}
    for key, value in grams_freq.items():
        if (key[0], key[1], key[2]) in dict_transitions_smooth:
            dict_transitions_smooth[(key[0], key[1], key[2])].update(
                {key[3]: (value + 1) / (
                        dict_start_cats_counts.get((key[0], key[1], key[2])) + total_categories_level2)})
        else:
            dict_transitions_smooth[(key[0], key[1], key[2])] = {
                key[3]: (value + 1) / (dict_start_cats_counts.get((key[0], key[1], key[2])) + total_categories_level2)}
    # print(dict_transitions_smooth)

    print("IN TRAIN SET:")
    # in train dataset:
    cases_right_train = 0
    nr_cases_train = 0
    for index, row in df_train.iterrows():
        nr_cases_train += 1
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        search_dict = dict_transitions_smooth.get((third_last_category, second_last_category, last_category))
        max_prob = 0
        predicted_cat_train = -1
        for key, val in search_dict.items():
            if val > max_prob:
                max_prob = val
                predicted_cat_train = key
        real_cat_train = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat_train, real_cat_train))
        if predicted_cat_train == real_cat_train:
            cases_right_train += 1
    accuracy_train = cases_right_train / nr_cases_train
    print("accuracy train = {}; nr cases gotten right in train set = {}; total nr of cases in train set = {}".format(
        accuracy_train, cases_right_train, nr_cases_train))

    print("IN VALIDATION SET:")
    # in validation dataset:
    cases_right_val = 0
    nr_cases_val = 0
    for index, row in df_validation.iterrows():
        nr_cases_val += 1
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if (third_last_category, second_last_category, last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get((third_last_category, second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_val += 1
    accuracy_validation = cases_right_val / nr_cases_val
    print("accuracy validation = {}; nr cases gotten right in val set = {}; total nr of cases in val set = {}".format(
        accuracy_validation, cases_right_val, nr_cases_val))

    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        if (third_last_category, second_last_category, last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get((third_last_category, second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_test += 1
    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))


print("****** 4-GRAMS *******")


# fourgrams()

def fivegrams():
    sequences_text = []
    grams_total = []
    for index, row in df_train.iterrows():
        sequence = row['sequence_cats_level2'] + [row['next_cat_level2']]
        sequences_text.extend(sequence)
        grams = ngrams(sequence, 5)
        grams_total += grams
    grams_freq = collections.Counter(grams_total)
    dict_start_cats_counts = {}
    for key, value in grams_freq.items():
        if (key[0], key[1], key[2], key[3]) in dict_start_cats_counts:
            dict_start_cats_counts[(key[0], key[1], key[2], key[3])] = dict_start_cats_counts.get(
                (key[0], key[1], key[2], key[3])) + value
        else:
            dict_start_cats_counts[(key[0], key[1], key[2], key[3])] = value
    dict_transitions_smooth = {}
    for key, value in grams_freq.items():
        if (key[0], key[1], key[2], key[3]) in dict_transitions_smooth:
            dict_transitions_smooth[(key[0], key[1], key[2], key[3])].update(
                {key[4]: (value + 1) / (
                        dict_start_cats_counts.get((key[0], key[1], key[2], key[3])) + total_categories_level2)})
        else:
            dict_transitions_smooth[(key[0], key[1], key[2], key[3])] = {
                key[4]: (value + 1) / (
                        dict_start_cats_counts.get((key[0], key[1], key[2], key[3])) + total_categories_level2)}
    # print(dict_transitions_smooth)

    print("IN TRAIN SET:")
    # in train dataset:
    cases_right_train = 0
    nr_cases_train = 0
    bla = 0
    for index, row in df_train.iterrows():
        nr_cases_train += 1
        fourth_last_category = row['sequence_cats_level2'][-4]
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        search_dict = dict_transitions_smooth.get(
            (fourth_last_category, third_last_category, second_last_category, last_category))
        max_prob = 0
        predicted_cat_train = -1
        for key, val in search_dict.items():
            if val > max_prob:
                max_prob = val
                predicted_cat_train = key
        real_cat_train = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat_train, real_cat_train))
        if predicted_cat_train == real_cat_train:
            cases_right_train += 1
    accuracy_train = cases_right_train / nr_cases_train
    print("accuracy train = {}; nr cases gotten right in train set = {}; total nr of cases in train set = {}".format(
        accuracy_train, cases_right_train, nr_cases_train))

    print("IN VALIDATION SET:")
    # in validation dataset:
    cases_right_val = 0
    nr_cases_val = 0
    for index, row in df_validation.iterrows():
        nr_cases_val += 1
        fourth_last_category = row['sequence_cats_level2'][-4]
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        if (fourth_last_category, third_last_category, second_last_category,
            last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get(
                (fourth_last_category, third_last_category, second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_val += 1
    accuracy_validation = cases_right_val / nr_cases_val
    print("accuracy validation = {}; nr cases gotten right in val set = {}; total nr of cases in val set = {}".format(
        accuracy_validation, cases_right_val, nr_cases_val))

    print("IN TEST SET:")
    cases_right_test = 0
    nr_cases_test = 0
    nr_else = 0
    for index, row in df_test.iterrows():
        nr_cases_test += 1
        fourth_last_category = row['sequence_cats_level2'][-4]
        third_last_category = row['sequence_cats_level2'][-3]
        second_last_category = row['sequence_cats_level2'][-2]
        last_category = row['sequence_cats_level2'][-1]
        # predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        if (fourth_last_category, third_last_category, second_last_category,
            last_category) in dict_transitions_smooth.keys():
            search_dict = dict_transitions_smooth.get(
                (fourth_last_category, third_last_category, second_last_category, last_category))
            max_prob = 0
            predicted_cat = -1
            for key, val in search_dict.items():
                if val > max_prob:
                    max_prob = val
                    predicted_cat = key
        else:
            nr_else += 1
            predicted_cat = unique_cats_level2[0]
        real_cat = row['next_cat_level2']
        # print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_test += 1
    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))
    print(nr_else)


print("****** 5-GRAMS *******")
# fivegrams()
