import ast
import os
import sys
from functools import partial
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from keras.layers import LSTM, Input, Masking, Dense, Concatenate
from keras.models import Sequential
from numpy import array
from keras.models import Model
import tensorflow as tf
import keras as keras
import keras.backend as K
from itertools import product
import itertools
from keras.callbacks import TensorBoard
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


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


def category_tree_to_get_parent(level_2_category):
    cat_tree_inverse = general_tree_category[0]
    return cat_tree_inverse[level_2_category]


def category_tree_to_get_siblings(level_2_category):
    parent = category_tree_to_get_parent(level_2_category)
    siblings = general_tree_category[1][parent]
    # siblings.remove(level_2_category)
    return siblings


# one hot encode value of sequence
def one_hot_encode(value, n_features, unique_cats):
    vector = [0 for _ in range(n_features)]
    vector[unique_cats.index(value)] = 1
    return vector


def siblings_encoded_array(siblings, n_features, unique_cats):
    vector = [0 for _ in range(n_features)]
    for value in siblings:
        vector[unique_cats.index(value)] = 1
    return vector


# decode a one hot encoded sequence
def one_hot_decode(encoded_seq, unique_cats):
    decoded_seq = []
    # returns index of highest probability value in array.
    for vector in encoded_seq:
        # print(vector)
        highest_value_index = np.argmax(vector)
        if vector[0] != -1:
            decoded_seq.append(unique_cats[highest_value_index])
    return decoded_seq


# decode a one hot encoded sequence
def decode_family(encoded_seq, unique_cats):
    decoded_seq = []
    # returns index of highest probability value in array.
    for vector in encoded_seq:
        highest_value_index = np.argmax(vector)
        if vector[0] != -1:
            decoded_seq.append(unique_cats[highest_value_index])
    return decoded_seq


def one_hot_decode_target(encoded_prediction, unique_cats):
    value = unique_cats[encoded_prediction.index(1)]
    return value


def lstm_model_categorical_data_concat():
    unique_cats_level2 = list(itertools.chain.from_iterable(general_tree_category[1].values()))
    unique_cats_level2.sort()
    print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))

    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    # df_train = df_train.loc[(df_train.visitorid == 404403)]
    # df_train = df_train.iloc[15:65]

    # '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set_temp.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    # train_pct_index = int(0.1 * len(df_train))
    # df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    # test_pct_index = int(0.1 * len(df_validation))
    # df_validation, df_discard_test = df_validation[:test_pct_index], df_validation[test_pct_index:]
    ############
    indexes_train = df_train.index.values
    indexes_validation = df_validation.index.values

    df_train['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level1']]
    df_validation['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level1']]
    df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level2']]
    df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level2']]
    df_train['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                   df_train['sequence_events']]
    df_validation['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                        df_validation['sequence_events']]

    X_train_level2 = df_train['sequence_cats_level2']
    y_train_level2 = df_train['next_cat_level2']
    X_val_level2 = df_validation['sequence_cats_level2']
    y_val_level2 = df_validation['next_cat_level2']

    # categories_level1 = len(unique_cats_level1)
    categories_level2 = len(unique_cats_level2)
    events_unique = 3
    timesteps = 0
    samples = X_train_level2.shape[0]
    print(X_train_level2.shape)
    samples_val = X_train_level2.shape[0]

    for current_sequence in X_train_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_level2.shape[0]
    for current_sequence in X_val_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    print("TIMESTEPS: " + str(timesteps))
    print("SAMPLES train: " + str(samples))
    print("SAMPLES val: " + str(samples_val))

    input_encoded_train = []
    output_encoded_train = []
    for x in indexes_train:
        one_sequence_encoding_input = []
        curr_sequence = X_train_level2[x]
        for v in range(len(curr_sequence)):
            value2 = X_train_level2[x][v]
            one_sequence_encoding_input.append(
                siblings_encoded_array(category_tree_to_get_siblings(value2), categories_level2, unique_cats_level2)
                + one_hot_encode(value2, categories_level2, unique_cats_level2))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append((([-1 for _ in range(categories_level2)]) +
                                                    ([-1 for _ in range(categories_level2)])))

        next_cat2 = y_train_level2[x]

        output_encoded_train.append(
            siblings_encoded_array(category_tree_to_get_siblings(next_cat2), categories_level2, unique_cats_level2) +
            one_hot_encode(next_cat2, categories_level2, unique_cats_level2))
        input_encoded_train.append(one_sequence_encoding_input)

    input_encoded_val = []
    output_encoded_val = []
    for z in indexes_validation:
        current_sequence = X_val_level2[z]
        one_sequence_encoding_input = []
        for v in range(len(current_sequence)):
            # value1 = X_val_level1[z][v]
            value2 = X_val_level2[z][v]
            one_sequence_encoding_input.append(  # (one_hot_encode(value1, categories_level1, unique_cats_level1) +
                siblings_encoded_array(category_tree_to_get_siblings(value2), categories_level2, unique_cats_level2) +
                one_hot_encode(value2, categories_level2, unique_cats_level2))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(([-1 for _ in range(categories_level2)] +
                                                    [-1 for _ in range(categories_level2)]))
        next_cat2 = y_val_level2[z]
        output_encoded_val.append(
            siblings_encoded_array(category_tree_to_get_siblings(next_cat2), categories_level2, unique_cats_level2) +
            one_hot_encode(next_cat2, categories_level2, unique_cats_level2))
        input_encoded_val.append(one_sequence_encoding_input)

    X_train = array(input_encoded_train)
    X_train = X_train.reshape(samples, timesteps, (categories_level2 + categories_level2))
    y_train = array(output_encoded_train)
    X_val = array(input_encoded_val)
    X_val = X_val.reshape(samples_val, timesteps, (categories_level2 + categories_level2))
    y_val = array(output_encoded_val)

    ################################ Evaluation metrics customization

    def categorical_accuracy_l2(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true[:, categories_level2:], axis=-1),
                              K.argmax(y_pred[:, categories_level2:], axis=-1)),
                      K.floatx())

    ################################        Loss Function Customization - Function Construction:

    def matrix_relations_construction():
        mat = np.zeros((categories_level2, categories_level2))
        for pred in range(categories_level2):
            for true in range(categories_level2):
                parent_pred = hierarchy_aux_dict[unique_cats_level2[pred]]
                parent_true = hierarchy_aux_dict[unique_cats_level2[true]]
                if parent_pred == parent_true:
                    mat[true, pred] = 1
                    mat[pred, true] = 1
        return mat

    def loss_function(matrix):
        matrix = tf.constant(matrix, np.float32)
        matrix_ones = np.ones((categories_level2, categories_level2))
        matrix_ones = tf.convert_to_tensor(matrix_ones, np.float32)
        print(matrix)

        def loss(y_true, y_pred):
            print(y_true[:, categories_level2:])
            print(y_pred[:, categories_level2:])

            l2_y_true = K.argmax(y_true[:, categories_level2:],
                                 axis=1)  # gets the index of the max value category in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample
            print(l2_y_true)
            l2_y_pred = K.argmax(y_pred[:, categories_level2:],
                                 axis=1)  # gets the index of the maximum prob category l2 predicted, in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample
            print(l2_y_pred)

            gather_mat = tf.gather_nd(matrix, tf.stack((l2_y_true, l2_y_pred), -1))
            print(gather_mat)

            cond = K.equal(gather_mat, K.constant(1))
            # K.equal(tf.gather_nd(matrix, [0, 1]), K.constant(1)) teste básico. tudo ok.
            return K.switch(cond,
                            # if the predicted l2 category is a sibling of the true l2 category -> turn the loss value lower by multiplying it by 0.1
                            lambda: K.categorical_crossentropy(y_true[:, categories_level2:],
                                                               y_pred[:, categories_level2:],
                                                               from_logits=False) * 0.1,
                            lambda: K.categorical_crossentropy(y_true[:, categories_level2:],
                                                               y_pred[:, categories_level2:],
                                                               from_logits=False))

        return loss

    def w_categorical_crossentropy_l2(y_true, y_pred):
        cross_ent = K.categorical_crossentropy(y_true[:, categories_level2:], y_pred[:, categories_level2:],
                                               from_logits=False)
        return cross_ent

    ################################    LSTM Model:
    def build_model(X_t, y_t, X_v, y_v, units, dropout, epochs, opt, loss, title):
        print("----")
        print("MODEL " + str(title))
        units = units
        dropout = dropout
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(timesteps, (categories_level2 + categories_level2))))
        model.add(LSTM(units, input_shape=(timesteps, (categories_level2 + categories_level2)), dropout=dropout,
                       return_sequences=False))  # units-> random number. trial and error methodology.
        model.add(Dense((categories_level2 + categories_level2), activation='softmax'))

        array_dummy = np.zeros((categories_level2, categories_level2))  # no relationships between categories here
        matrix_siblings_relations = matrix_relations_construction()
        if loss != 'standard':
            model.compile(loss=loss_function(matrix_siblings_relations), optimizer=opt,
                          metrics=[categorical_accuracy_l2])
        else:
            custom_loss = partial(w_categorical_crossentropy_l2)
            custom_loss.__name__ = 'w_categorical_crossentropy'
            model.compile(loss=custom_loss, optimizer=opt,
                          metrics=[categorical_accuracy_l2])

        # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
        print(model.summary())
        print("model fit")
        history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=epochs, verbose=0)
        print("model evaluate")
        print(model.metrics_names)
        loss_train, acc_train = model.evaluate(X_t, y_t, verbose=0)
        # print('loss_train: %f' % (loss_train * 100))
        print('acc_train: %f' % (acc_train * 100))

        loss_val, acc_val = model.evaluate(X_v, y_v, verbose=0)
        #print('loss_val: %f' % (loss_val * 100))
        print('acc_val: %f' % (acc_val * 100))

        # plt.plot(history.history['categorical_accuracy_l2']) #history verificar se aqui nao é devolvido validation acc tbm.
        # plt.show()
        '''
        ################################## experiments
        X_test_l2 = df_train.loc[(df_train.visitorid == 404403)]
        X_test = X_test_l2['sequence_cats_level2']
        y_test = X_test['next_cat_level2']
        testing_samples_sequences = X_test  # X_train_level2
        testing_samples_next_category = y_test  # y_train_level2
        gotten_right = 0
        gotten_wrong = 0
        for i in range(len(testing_samples_sequences)):
            sample_sequence = testing_samples_sequences.iloc[i]
            encoded_test_seq_input = []
            for pred in range(len(sample_sequence)):
                v2 = sample_sequence[pred]
                encoded_test_seq_input.append(
                    siblings_encoded_array(category_tree_to_get_siblings(v2), categories_level2, unique_cats_level2) +
                    one_hot_encode(v2, categories_level2, unique_cats_level2))

            if (pred + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
                for true in range(timesteps - (pred + 1)):
                    encoded_test_seq_input.append(([-1 for _ in range(categories_level2)] +
                                                   [-1 for _ in range(categories_level2)]))

            X_test_sample = [encoded_test_seq_input]
            X_test_sample = array(X_test_sample)
            # print(X_test_sample.shape)

            y_test_real_level2 = testing_samples_next_category.iloc[i]

            y_predicted = model.predict(X_test_sample)
            encoded_test_l2 = []
            encoded_test_family_l2 = []
            for position_seq in X_test_sample[0]:
                encoded_complete_value = position_seq
                encoded_family_level2 = encoded_complete_value[:categories_level2]
                encoded_test_family_l2.append(encoded_family_level2.tolist())
                encoded_value_level2 = encoded_complete_value[categories_level2:]
                encoded_test_l2.append(encoded_value_level2.tolist())

            decoded_value_level2 = one_hot_decode(encoded_test_l2, unique_cats_level2)
            print("----")
            print("Sequence level2: %s" % decoded_value_level2)
            print('Expected level2: %s' % y_test_real_level2)

            pred_l2 = y_predicted[0][categories_level2:]
            pred_family_l2 = y_predicted[0][:categories_level2]
            # print("START")
            # print(one_hot_decode([pred_family_l2], unique_cats_level2)[0])
            # print('Predicted level2: %s' % [pred_l2], unique_cats_level2[0])
            # print("END")
            print('Predicted decoded level2: %s' % one_hot_decode([pred_l2], unique_cats_level2)[0])

            if one_hot_decode([pred_l2], unique_cats_level2)[0] == y_test_real_level2:
                gotten_right += 1
            else:
                gotten_wrong += 1
                print("*--------> wrong prediction!")
        print("nr of right predictions = %s" % gotten_right)
        print("nr of wrong predictions = %s" % gotten_wrong)
        '''
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    build_model(X_train, y_train, X_val, y_val, 30, 0.2, 150, 'adagrad', 'standard',
                'standard categorical crossentropy loss')
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    build_model(X_train, y_train, X_val, y_val, 30, 0.2, 150, 'adagrad', 'custom', 'custom loss')

    ##################################


print("******************************* MODEL W/ INPUT INSTANCES W/ ENCODING OF CATEGORY L2 AND ITS SIBLINGS")
lstm_model_categorical_data_concat()


def lstm_model_categorical_data_first_l2():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')
    unique_cats_level1 = list(df_visitors_no_split['level1'].unique())
    unique_cats_level1.sort()
    # print(unique_cats_level1)
    # print("Distinct nr of categories level1: %s" % (len(unique_cats_level1)))
    unique_cats_level2 = list(df_visitors_no_split['level2'].unique())
    unique_cats_level2.sort()
    # print(unique_cats_level2)
    # print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))

    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    df_train = df_train.loc[(df_train.visitorid == 404403)]
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    # train_pct_index = int(0.1 * len(df_train))
    # df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    # test_pct_index = int(0.1 * len(df_validation))
    # df_validation, df_discard_test = df_validation[:test_pct_index], df_validation[test_pct_index:]
    ############
    indexes_train = df_train.index.values

    df_train['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level1']]
    df_validation['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level1']]
    df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level2']]
    df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level2']]
    df_train['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                   df_train['sequence_events']]
    df_validation['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                        df_validation['sequence_events']]

    X_train_level2 = df_train['sequence_cats_level2']
    y_train_level2 = df_train['next_cat_level2']

    X_val_level2 = df_validation['sequence_cats_level2']
    y_val_level2 = df_validation['next_cat_level2']

    categories_level2 = len(unique_cats_level2)
    timesteps = 0
    samples = X_train_level2.shape[0]
    samples_val = X_val_level2.shape[0]

    for current_sequence in X_train_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_level2.shape[0]
    for current_sequence in X_val_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    print("TIMESTEPS: " + str(timesteps))
    print("SAMPLES train: " + str(samples))
    print("SAMPLES val: " + str(samples_val))

    input_encoded_train = []
    output_encoded_train = []

    for x in indexes_train:
        one_sequence_encoding_input = []
        current_sequence = X_train_level2[x]
        for v in range(len(current_sequence)):
            value2 = X_train_level2[x][v]
            one_sequence_encoding_input.append(
                one_hot_encode(value2, categories_level2, unique_cats_level2))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(
                    ([-1 for _ in range(categories_level2)]))

        next_cat2 = y_train_level2[x]
        output_encoded_train.append(
            one_hot_encode(next_cat2, categories_level2, unique_cats_level2))
        input_encoded_train.append(one_sequence_encoding_input)

    input_encoded_val = []
    output_encoded_val = []
    for z in range(samples_val):
        current_sequence = X_val_level2[z]
        one_sequence_encoding_input = []
        for v in range(len(current_sequence)):
            value2 = X_val_level2[z][v]
            one_sequence_encoding_input.append(
                one_hot_encode(value2, categories_level2, unique_cats_level2))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(
                    [-1 for _ in range(categories_level2)])
        next_cat2 = y_val_level2[z]
        output_encoded_val.append(
            one_hot_encode(next_cat2, categories_level2, unique_cats_level2))
        input_encoded_val.append(one_sequence_encoding_input)

    X_train = array(input_encoded_train)
    X_train = X_train.reshape(samples, timesteps, categories_level2)
    y_train = array(output_encoded_train)
    X_val = array(input_encoded_val)
    X_val = X_val.reshape(samples_val, timesteps, categories_level2)
    y_val = array(output_encoded_val)

    ######## acc metric
    def categorical_accuracy_l2(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true, axis=-1),
                              K.argmax(y_pred, axis=-1)),
                      K.floatx())

    ################################        Loss Function Customization - Function Construction:
    def w_categorical_crossentropy_none(y_true, y_pred):
        cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
        return cross_ent

    def matrix_relations_construction():
        mat = np.zeros((categories_level2, categories_level2))
        for pred in range(categories_level2):
            for true in range(categories_level2):
                parent_pred = hierarchy_aux_dict[unique_cats_level2[pred]]
                parent_true = hierarchy_aux_dict[unique_cats_level2[true]]
                if parent_pred == parent_true:
                    mat[true, pred] = 1
                    mat[pred, true] = 1
        return mat

    def loss_function(matrix):
        matrix = tf.constant(matrix, np.float32)
        print(matrix)

        def loss(y_true, y_pred):
            print(y_true)
            print(y_pred)

            l2_y_true = K.argmax(y_true[:, :],
                                 axis=1)  # gets the index of the max value category in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample
            l2_y_pred = K.argmax(y_pred[:, :],
                                 axis=1)  # gets the index of the maximum prob category l2 predicted, in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample

            gather_mat = tf.gather_nd(matrix, tf.stack((l2_y_true, l2_y_pred), -1))

            cond = K.equal(gather_mat, K.constant(1))
            return K.switch(cond,
                            # if the predicted l2 category is a sibling of the true l2 category -> turn the loss value lower by multiplying it by 0.1
                            lambda: K.categorical_crossentropy(y_true[:, :],
                                                               y_pred[:, :],
                                                               from_logits=False) * 0.1,
                            lambda: K.categorical_crossentropy(y_true[:, :],
                                                               y_pred[:, :],
                                                               from_logits=False))

        return loss

    ################################    LSTM Model:
    def build_model(X_t, y_t, X_v, y_v, units, dropout, epochs, opt, loss, title):
        metric_l2 = partial(categorical_accuracy_l2)
        metric_l2.__name__ = 'categorical_accuracyl2'
        print("MODEL " + str(title))
        units = units
        dropout = dropout
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(timesteps, categories_level2)))
        model.add(LSTM(units, input_shape=(timesteps, categories_level2), dropout=dropout,
                       return_sequences=False))
        model.add(Dense(categories_level2, activation='softmax'))
        if loss == 'standard':
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=['acc'])
        else:
            matrix_siblings_relations = matrix_relations_construction()
            model.compile(loss=loss_function(matrix_siblings_relations), optimizer=opt,
                          metrics=['acc'])
        print(model.summary())
        # history = LossHistory()
        print("model fit")
        history = model.fit(X_t, y_t, epochs=epochs,
                            verbose=0)  # validation_data=(X_v, y_v), epochs=epochs, verbose=0)
        print("model evaluate")
        print(model.metrics_names)

        loss_train, accuracy_train = model.evaluate(X_t, y_t, verbose=0)
        print('Accuracy train: %f' % (accuracy_train * 100))
        # loss_val, accuracy_val = model.evaluate(X_v, y_v, verbose=0)
        # print('Accuracy validation: %f' % (accuracy_val * 100))

        '''if loss == 'standard':
            # PLOT
            plt.ylim(0.0, 1.0)
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.title(title, pad=-80)

            # plt.legend(['train', 'validation'], loc='upper left')
            # plt.show()'''

        ################################## experiments

        testing_samples_sequences = X_train_level2
        testing_samples_next_category = y_train_level2
        gotten_right = 0
        gotten_wrong = 0
        for i in range(len(testing_samples_sequences)):
            sample_sequence = testing_samples_sequences.iloc[i]
            encoded_test_seq_input = []
            for pred in range(len(sample_sequence)):
                v2 = sample_sequence[pred]
                encoded_test_seq_input.append(
                    one_hot_encode(v2, categories_level2, unique_cats_level2))

            if (pred + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
                for true in range(timesteps - (pred + 1)):
                    encoded_test_seq_input.append((
                        [-1 for _ in range(categories_level2)]))

            X_test_sample = [encoded_test_seq_input]
            X_test_sample = array(X_test_sample)
            # print(X_test_sample.shape)

            y_test_real_level2 = testing_samples_next_category.iloc[i]

            y_predicted = model.predict(X_test_sample)
            encoded_test_l2 = []
            for position_seq in X_test_sample[0]:
                encoded_complete_value = position_seq
                encoded_value_level2 = encoded_complete_value[:]
                encoded_test_l2.append(encoded_value_level2.tolist())

            decoded_value_level2 = one_hot_decode(encoded_test_l2, unique_cats_level2)
            print("----")
            print("Sequence level2: %s" % decoded_value_level2)
            print('Expected level2: %s' % y_test_real_level2)
            pred_l2 = y_predicted[0][:]

            print('Predicted decoded level2: %s' % one_hot_decode([pred_l2], unique_cats_level2)[0])

            if one_hot_decode([pred_l2], unique_cats_level2)[0] == y_test_real_level2:
                gotten_right += 1
            else:
                gotten_wrong += 1
                print("*--------> wrong prediction!")
        print("nr of right predictions = %s" % gotten_right)
        print("nr of wrong predictions = %s" % gotten_wrong)

    '''
    momentums = ['sgd', 'rmsprop', 'adagrad', 'adam']
    dropouts = [0, 0.2, 0.4, 0.6]
    units = [10, 30, 50, 70]
    epochs = [50, 100, 200, 300]
    for i in range(len(units)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        plot_no = 220 + (i + 1)
        plt.subplot(plot_no)
        # fit model and plot learning curves for an optimizer
        build_model(X_train, y_train, X_val, y_val, 40, 0.2, epochs[i], 'adagrad', 'standard', epochs[i], 'none', 0)
    '''
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    build_model(X_train, y_train, X_val, y_val, 30, 0.2, 150, 'adagrad', 'standard',
                'standard categorical accuracy keras')
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    build_model(X_train, y_train, X_val, y_val, 30, 0.2, 150, 'adagrad', 'custom',
                'standard categorical accuracy keras')

    # build_model(X_train, y_train, X_val, y_val, 30, 0, 100, 'adagrad', w_categorical_crossentropy_none,
    #           'categorical accuracy keras backend, predict l2', 'none', 0)

    # plt.show()


print("******************************* MODEL W/ INPUT INSTANCES W/ JUST 1-HOT ENCODING CATEGORY L2")


# lstm_model_categorical_data_first_l2()


##### STANDARD LSTM MODEL. FIRST MODEL:
def lstm_model_categorical_data_first_l1():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')
    unique_cats_level1 = list(df_visitors_no_split['level1'].unique())
    unique_cats_level1.sort()
    print(unique_cats_level1)
    print("Distinct nr of categories level1: %s" % (len(unique_cats_level1)))

    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    train_pct_index = int(0.1 * len(df_train))
    df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    test_pct_index = int(0.3 * len(df_validation))
    df_validation, df_discard_test = df_validation[:test_pct_index], df_validation[test_pct_index:]
    ############
    df_train['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level1']]
    df_validation['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level1']]
    df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level2']]
    df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level2']]
    df_train['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                   df_train['sequence_events']]
    df_validation['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                        df_validation['sequence_events']]

    X_train_level1 = df_train['sequence_cats_level1']
    y_train_level1 = df_train['next_cat_level1']

    X_val_level1 = df_validation['sequence_cats_level1']
    y_val_level1 = df_validation['next_cat_level1']

    categories_level1 = len(unique_cats_level1)
    timesteps = 0
    samples = X_train_level1.shape[0]
    samples_val = X_val_level1.shape[0]

    for current_sequence in X_train_level1:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_level1.shape[0]
    for current_sequence in X_val_level1:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    print("TIMESTEPS: " + str(timesteps))
    print("SAMPLES train: " + str(samples))
    print("SAMPLES val: " + str(samples_val))

    input_encoded_train = []
    output_encoded_train = []
    for x in range(samples):
        one_sequence_encoding_input = []
        current_sequence = X_train_level1[x]
        for v in range(len(current_sequence)):
            value2 = X_train_level1[x][v]
            one_sequence_encoding_input.append(
                one_hot_encode(value2, categories_level1, unique_cats_level1))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(
                    ([-1 for _ in range(categories_level1)]))

        next_cat2 = y_train_level1[x]
        output_encoded_train.append(
            one_hot_encode(next_cat2, categories_level1, unique_cats_level1))
        input_encoded_train.append(one_sequence_encoding_input)

    input_encoded_val = []
    output_encoded_val = []
    for z in range(samples_val):
        current_sequence = X_val_level1[z]
        one_sequence_encoding_input = []
        for v in range(len(current_sequence)):
            value2 = X_val_level1[z][v]
            one_sequence_encoding_input.append(
                one_hot_encode(value2, categories_level1, unique_cats_level1))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(
                    [-1 for _ in range(categories_level1)])
        next_cat2 = y_val_level1[z]
        output_encoded_val.append(
            one_hot_encode(next_cat2, categories_level1, unique_cats_level1))
        input_encoded_val.append(one_sequence_encoding_input)

    X_train = array(input_encoded_train)
    X_train = X_train.reshape(samples, timesteps, categories_level1)
    y_train = array(output_encoded_train)
    X_val = array(input_encoded_val)
    X_val = X_val.reshape(samples_val, timesteps, categories_level1)
    y_val = array(output_encoded_val)

    ################################        Loss Function Customization - Function Construction:
    def w_categorical_crossentropy_none(y_true, y_pred):
        cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=True)
        return cross_ent

    ################################       Loss Function Customization - Weights Matrices
    ## A) dummy weigths matrix construction. Matrix w/ all 1s. All categories relationships w/ the same importance:
    w_array_dummy = np.ones((categories_level1, categories_level1))

    print("weights matrix constructed.")

    ################################    LSTM Model:
    def build_model(X_t, y_t, X_v, y_v, units, dropout, epochs, opt, loss, title, weights_mat, flag_weight):

        print("MODEL " + str(title))
        units = units
        dropout = dropout
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(timesteps, categories_level1)))
        model.add(LSTM(units, input_shape=(timesteps, categories_level1), dropout=dropout,
                       return_sequences=False))  # units-> random number. trial and error methodology.
        model.add(Dense(categories_level1, activation='softmax'))
        if loss == 'standard':
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=['acc'])
        else:
            if flag_weight == 1:
                custom_loss = partial(loss, weights=weights_mat)
            else:
                custom_loss = partial(loss)
            custom_loss.__name__ = 'w_categorical_crossentropy'
            model.compile(loss=custom_loss, optimizer=opt, metrics=['acc'])
        print(model.summary())
        # history = LossHistory()
        print("model fit")
        history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=epochs, verbose=0)
        print("model evaluate")
        print(model.metrics_names)

        loss_train, accuracy_train = model.evaluate(X_v, y_v, verbose=0)
        print('Accuracy train: %f' % (accuracy_train * 100))
        loss_val, accuracy_val = model.evaluate(X_v, y_v, verbose=0)
        print('Accuracy validation: %f' % (accuracy_val * 100))

        # PLOT
        plt.ylim(0.0, 1.0)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.title(opt, pad=-80)

        # plt.legend(['train', 'validation'], loc='upper left')
        # plt.show()'''

        ################################## test on a new sample. prediction on a SINGLE new data sequence. experiment
        sequence_test_level2 = [653, 1490, 1490, 653]
        encoded_test_seq_input = []
        for pred in range(len(sequence_test_level2)):
            v2 = sequence_test_level2[pred]
            encoded_test_seq_input.append(
                one_hot_encode(v2, categories_level1, unique_cats_level1))

        if (pred + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for true in range(timesteps - (pred + 1)):
                encoded_test_seq_input.append((
                    [-1 for _ in range(categories_level1)]))

        X_test_sample = [encoded_test_seq_input]
        X_test_sample = array(X_test_sample)
        # print(X_test_sample.shape)

        y_test_real_level1 = 653

        y_predicted = model.predict(X_test_sample)
        encoded_test_l1 = []
        for position_seq in X_test_sample[0]:
            encoded_complete_value = position_seq
            encoded_value_level1 = encoded_complete_value[:]
            encoded_test_l1.append(encoded_value_level1.tolist())

        decoded_value_level1 = one_hot_decode(encoded_test_l1, unique_cats_level1)
        print("Sequence level2: %s" % decoded_value_level1)
        print('Expected level2: %s' % y_test_real_level1)

        pred_l1 = y_predicted[0][:]

        # print('Predicted level2: %s' % [pred_l2], unique_cats_level2[0])

        print('Predicted decoded level2: %s' % one_hot_decode([pred_l1], unique_cats_level1)[0])

    momentums = ['sgd', 'rmsprop', 'adagrad', 'adam']
    dropouts = [0, 0.2, 0.4, 0.6]
    units = [10, 30, 50, 70]
    for i in range(len(units)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        plot_no = 220 + (i + 1)
        plt.subplot(plot_no)
        # fit model and plot learning curves for an optimizer
        build_model(X_train, y_train, X_val, y_val, 30, 0, 100, momentums[i], 'standard',
                    momentums[i], 'none', 0)

    # build_model(X_train, y_train, X_val, y_val, 30, 0, 100, 'adagrad', 'standard', 'standard categorical accuracy keras', 'none', 0)
    # build_model(X_train, y_train, X_val, y_val, 30, 0, 100, 'adagrad', w_categorical_crossentropy_none, 'categorical accuracy keras backend, predict l2', 'none', 0)

    plt.show()

# print('Sequence: %s' % [one_hot_decode(X_test_new[0][i]) for i in range(len(X_test_new))])
# print('Expected: %s' % y_test_new)
# print('Predicted: %s' % one_hot_decode(y_predicted))


# lstm_model_categorical_data_first_l1()
