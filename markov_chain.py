import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
from pprint import pprint
import ast

pd.options.display.max_columns
import itertools

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


# print(dict_categories)

def init_counts_matrix(unique_cats_level2, total_categories_level2):
    # create initial matrix, with all zeroes.
    trans_matrix = np.zeros([total_categories_level2, total_categories_level2],
                            dtype=float)  # creating initial matrix of all-zeros
    dict_categories = dict.fromkeys(unique_cats_level2, 0)

    return trans_matrix, dict_categories


def update_counts_matrix(unique_cats_level2, trans_matrix, dict_categories_counts, sequence):
    for (i, j) in zip(sequence, sequence[1:]):  # transition from one category to another.
        index_i = unique_cats_level2.index(i)
        index_j = unique_cats_level2.index(j)
        trans_matrix[index_i][index_j] += 1
        dict_categories_counts[i] += 1  # count number of times it starts from i

    '''for r in trans_matrix:
        print(' '.join('{}'.format(x) for x in r))'''


def create_transitions_matrix(matrix_counts, dict_categories_counts, unique_cats_level2, total_categories_level2):
    matrix_probs = matrix_counts.copy()
    # applying Laplace smoothing here:
    for i in range(len(matrix_probs)):
        for j in range(len(matrix_probs[i])):
            # print(list_indexes[i])
            matrix_probs[i][j] = (matrix_probs[i][j] + 1) / (
                    dict_categories_counts[unique_cats_level2[i]] + total_categories_level2)

    '''for r in matrix_probs:
        print(' '.join('{0:.5f}'.format(x) for x in r))'''

    return matrix_probs


def train_validate_model():
    unique_cats_level2 = list(itertools.chain.from_iterable(general_tree_category[1].values()))
    unique_cats_level2.sort()
    print(unique_cats_level2)
    print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))
    df_train = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_test_set.csv')
    df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level2']]
    df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level2']]
    df_test['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                       df_test['sequence_cats_level2']]

    total_categories_level2 = len(unique_cats_level2)
    counts_matrix, dict_counts = init_counts_matrix(unique_cats_level2, total_categories_level2)
    for index, row in df_train.iterrows():
        sequence = row['sequence_cats_level2'] + [row['next_cat_level2']]
        update_counts_matrix(unique_cats_level2, counts_matrix, dict_counts, sequence)
    transitions_matrix = create_transitions_matrix(counts_matrix, dict_counts, unique_cats_level2,
                                                   total_categories_level2)
    print("IN TRAIN SET:")
    # in train dataset:
    cases_right_train = 0
    nr_cases_train = 0
    for index, row in df_train.iterrows():
        nr_cases_train += 1
        last_category = row['sequence_cats_level2'][-1]
        predicted_cat_train = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        real_cat_train = row['next_cat_level2']
        print("predicted next category: {}; real next category: {}".format(predicted_cat_train, real_cat_train))
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
        predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        real_cat = row['next_cat_level2']
        print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
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
        last_category = row['sequence_cats_level2'][-1]
        predicted_cat = unique_cats_level2[np.argmax(transitions_matrix[unique_cats_level2.index(last_category)])]
        real_cat = row['next_cat_level2']
        print("predicted next category: {}; real next category: {}".format(predicted_cat, real_cat))
        if predicted_cat == real_cat:
            cases_right_test += 1
    accuracy_test = cases_right_test / nr_cases_test
    print("accuracy test = {}; nr cases gotten right in test set = {}; total nr of cases in test set = {}".format(
        accuracy_test, cases_right_test, nr_cases_test))


train_validate_model()
