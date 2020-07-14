import ast
import os
import sys
from functools import partial

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
    tree_dict = empty_initial_categories_dict()  ###level 1
    list_root_categories = list(tree_dict.keys())
    aux_dict_2 = {}
    for cat in list_root_categories:  ###level 2
        categories_next_level = df_cat.loc[(df_cat.parentid == int(cat))]
        l = list(categories_next_level.categoryid)
        l.sort()
        tree_dict[cat] = l
        for elem in l:
            aux_dict_2[elem] = cat
    return aux_dict_2

# one hot encode value of sequence
def one_hot_encode(value, n_features, unique_cats):
    vector = [0 for _ in range(n_features)]
    vector[unique_cats.index(value)] = 1
    return vector

# decode a one hot encoded sequence
def one_hot_decode(encoded_seq, unique_cats):
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
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    train_pct_index = int(0.25 * len(df_train))
    df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    test_pct_index = int(0.10 * len(df_validation))
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
    X_train_level2 = df_train['sequence_cats_level2']
    y_train_level2 = df_train['next_cat_level2']
    X_val_level1 = df_validation['sequence_cats_level1']
    y_val_level1 = df_validation['next_cat_level1']
    X_val_level2 = df_validation['sequence_cats_level2']
    y_val_level2 = df_validation['next_cat_level2']

    categories_level1 = len(unique_cats_level1)
    categories_level2 = len(unique_cats_level2)
    events_unique = 3
    timesteps = 0
    units = 25
    samples = X_train_level1.shape[0]

    for current_sequence in X_train_level1:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_level1.shape[0]
    for current_sequence in X_val_level1:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    hierarchy_aux_dict = category_tree()  # for the weights matrix (easier level1 parent search) to be used in the loss function.

    input_encoded_train = []
    output_encoded_train = []
    for x in range(samples):
        one_sequence_encoding_input = []
        current_sequence = X_train_level1[x]
        for v in range(len(current_sequence)):
            value1 = X_train_level1[x][v]
            value2 = X_train_level2[x][v]
            one_sequence_encoding_input.append((one_hot_encode(value1, categories_level1, unique_cats_level1) +
                                                one_hot_encode(value2, categories_level2, unique_cats_level2)))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append((([-1 for _ in range(categories_level1)]) +
                                                    ([-1 for _ in range(categories_level2)])))

        next_cat1 = y_train_level1[x]
        next_cat2 = y_train_level2[x]

        output_encoded_train.append((one_hot_encode(next_cat1, categories_level1, unique_cats_level1) +
                                     one_hot_encode(next_cat2, categories_level2, unique_cats_level2)))
        input_encoded_train.append(one_sequence_encoding_input)

    input_encoded_val = []
    output_encoded_val = []
    for z in range(samples_val):
        current_sequence = X_val_level1[z]
        one_sequence_encoding_input = []
        for v in range(len(current_sequence)):
            value1 = X_val_level1[z][v]
            value2 = X_val_level2[z][v]
            one_sequence_encoding_input.append((one_hot_encode(value1, categories_level1, unique_cats_level1) +
                                                one_hot_encode(value2, categories_level2, unique_cats_level2)))
        if (v + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for real in range(timesteps - (v + 1)):
                one_sequence_encoding_input.append(([-1 for _ in range(categories_level1)] +
                                                    [-1 for _ in range(categories_level2)]))
        next_cat1 = y_val_level1[z]
        next_cat2 = y_val_level2[z]
        output_encoded_val.append((one_hot_encode(next_cat1, categories_level1, unique_cats_level1) +
                                   one_hot_encode(next_cat2, categories_level2, unique_cats_level2)))
        input_encoded_val.append(one_sequence_encoding_input)

    X_train = array(input_encoded_train)
    X_train = X_train.reshape(samples, timesteps, (categories_level1 + categories_level2))
    y_train = array(output_encoded_train)
    X_val = array(input_encoded_val)
    X_val = X_val.reshape(samples_val, timesteps, (categories_level1 + categories_level2))
    y_val = array(output_encoded_val)

    ################################        Loss Function Customization - Example w/ numpy:
    print("EXAMPLE CASE LOSS FUNCTION W/ WEIGHTS")

    y_true_example = array([[0, 1, 0, 0, 1, 0],
                            [1, 0, 0, 0, 0, 1],
                            [1, 0, 0, 0, 0, 1]])
    y_pred_example = array(
        [[0.01, 0.8, 0.3, 0.1, 0.1, 0.0],
         [0.7, 0.4, 0.5, 0.0, 0.25, 0.6],
         [0.3, 0.2, 0.2, 0.02, 0.3, 0.7]])

    weights_example = array([[1, 0.3, 0.3, 0.3, 0.8, 0.1],
                             [0.3, 1, 0.3, 0.3, 0.1, 0.8],
                             [0.3, 0.3, 1, 0.3, 0.8, 0.8],
                             [0.3, 0.3, 0.3, 1, 0.8, 0.8],
                             [0.8, 0.1, 0.8, 0.8, 1, 0.3],
                             [0.1, 0.8, 0.8, 0.8, 0.3, 1]])

    y_pred_max1_example = np.max(y_pred_example[:, :4],
                                 axis=1)  # selects the feature level1 w/ the maximum value in the sample.
    y_pred_max2_example = np.max(y_pred_example[:, 4:],
                                 axis=1)  # selects the feature level2 w/ the maximum value in the sample.

    y_pred_max1_example = np.expand_dims(y_pred_max1_example, 1)
    y_pred_max2_example = np.expand_dims(y_pred_max2_example, 1)

    y_pred_max_mat1 = np.equal(y_pred_example[:, :4], y_pred_max1_example)
    y_pred_max_mat2 = np.equal(y_pred_example[:, 4:], y_pred_max2_example)

    y_pred_max_mat_example = np.concatenate((y_pred_max_mat1, y_pred_max_mat2), axis=1)

    final_mask_example = np.zeros_like(y_pred_example[:, 0])  # array([0, 0, 0])

    for predicted, true in product(range(len(weights_example)), range(len(weights_example))):
        print('\n')
        print("-----> predicted, current: " + str(predicted) + "\t" + str(true))
        print("weights_example[predicted, true] = " + str(weights_example[predicted, true]))
        print("y_pred_max_mat[:, predicted] = " + str(y_pred_max_mat_example[:, predicted]))
        print("y_true_example[:, true] = " + str(y_true_example[:, true]))
        print(str(weights_example[predicted, true]) + " * " + str(y_pred_max_mat_example[:, predicted]) + " * " + str(
            y_true_example[:, true]) + " = "
              + str(
            (weights_example[predicted, true] * y_pred_max_mat_example[:, predicted] * y_true_example[:, true])))
        final_mask_example += (
                weights_example[predicted, true] * y_pred_max_mat_example[:, predicted] * y_true_example[:, true])
        print("final mask current: " + str(final_mask_example))
    print('\n')
    print("final mask: " + str(final_mask_example))

    ################################        Loss Function Customization - Function Construction:

    def w_categorical_crossentropy(y_true, y_pred, weights):
        print(y_true.shape)  # it has to have the same shape as y_train.
        # [[1-hot encoding next category in seq1], [1-hot encoding next category in seq2], [1-hot encoding next category in seq3], ...]
        # so, (samples, total features). it prints: (None, None).
        print(y_pred.shape)  # again, (samples, total features).
        # it prints: (None, 151). note: 151 is the total number of features.
        weights_matrix_len = len(weights)  # it prints: 151

        y_pred_max = K.max(y_pred,
                           axis=1)  # gets the maximum probability from each sample. # [max_prob in sample1, max_prob in sample2, max_prob in sample3,...] #printed shape: (None,).

        y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))

        y_pred_max_mat = K.cast(K.equal(y_pred, y_pred_max), K.floatx())
        # K.equal(y_pred, y_pred_max) returns a boolean tensor,
        # convert a boolean tensor into a float32 tensor type.
        # prints shape=(None, 151).

        final_mask = K.zeros_like(y_pred[:, 0])  # each row of y_pred with the value of 0.
        # array [0, 0, 0, 0, 0, ...] total of samples
        # initial mask to be used next when constructing the mask in the loop.

        for pos_pred, pos_true in product(range(weights_matrix_len), range(
                weights_matrix_len)):  # product python. goes through the weights square matrix (weights_matrix_len X weights_matrix_len)
            final_mask += (weights[pos_pred, pos_true] * y_pred_max_mat[:, pos_pred] * y_true[:, pos_true])
        # in the end, the final mask has the adjusted weight for each sample in the dataset.
        cross_ent = K.categorical_crossentropy(y_true, y_pred, from_logits=False)
        return cross_ent * final_mask

    def w_categorical_crossentropy_levels(y_true, y_pred, weights):
        weights_matrix_len = len(weights)

        y_pred_max_l1 = K.max(y_pred[:, :categories_level1],
                              axis=1)
        y_pred_max_l2 = K.max(y_pred[:, categories_level1:],
                              axis=1)
        y_pred_max_l1 = K.reshape(y_pred_max_l1, (K.shape(y_pred)[0], 1))
        y_pred_max_l2 = K.reshape(y_pred_max_l2, (K.shape(y_pred)[0], 1))

        y_pred_max_mat_l1 = K.cast(K.equal(y_pred[:, :categories_level1], y_pred_max_l1), K.floatx())
        y_pred_max_mat_l2 = K.cast(K.equal(y_pred[:, categories_level1:], y_pred_max_l2), K.floatx())

        y_pred_max_mat = K.concatenate((y_pred_max_mat_l1, y_pred_max_mat_l2), axis=1)

        final_mask = K.zeros_like(y_pred[:, 0])

        for pos_pred, pos_true in product(range(weights_matrix_len), range(
                weights_matrix_len)):
            final_mask += (weights[pos_pred, pos_true] * y_pred_max_mat[:, pos_pred] * y_true[:, pos_true])

        cross_ent_1 = K.categorical_crossentropy(y_true[:, :categories_level1], y_pred[:, :categories_level1],
                                                 from_logits=False)
        cross_ent_2 = K.categorical_crossentropy(y_true[:, categories_level1:], y_pred[:, categories_level1:],
                                                 from_logits=False)

        cross_ent = cross_ent_1 + cross_ent_2
        return cross_ent * final_mask

    ################################       Loss Function Customization - Weights Matrices
    ## A) dummy weigths matrix construction:
    w_array_dummy = np.ones(((categories_level1 + categories_level2), (categories_level1 + categories_level2)))

    ## B) Weigths matrix construction:
    w_array_basic = np.ones(((categories_level1 + categories_level2), (categories_level1 + categories_level2)))
    high_weight = 0.8
    medium_weight = 0.3

    for predicted in range((categories_level1 + categories_level2)):
        for real in range((categories_level1 + categories_level2)):
            if predicted != real:
                if predicted < categories_level1 and real > categories_level1:  # predicted category of level1 when it should be of level2
                    w_array_basic[predicted, real] = high_weight
                elif predicted < categories_level1 and real < categories_level1:
                    w_array_basic[predicted, real] = medium_weight
                elif predicted > categories_level1 and real < categories_level1:
                    w_array_basic[predicted, real] = high_weight
                elif predicted > categories_level1 and real > categories_level1:
                    w_array_basic[predicted, real] = medium_weight

    ## C) Weigths matrix real costumization:
    w_array_new = np.ones(((categories_level1 + categories_level2), (categories_level1 + categories_level2)))
    high_weight_new = 0.8  # higher penalization
    medium_weight_new = 0.3  # medium penalization
    low_weight_new = 0.1  # low penalization
    for predicted in range((categories_level1 + categories_level2)):
        for real in range((categories_level1 + categories_level2)):
            if predicted != real:
                if predicted > categories_level1 and real < categories_level1: #when predicted and real categories are from different levels. establish here a relationship (a strong one or weak one).
                    predicted_l2 = predicted
                    value_level2 = unique_cats_level2[predicted_l2-categories_level1]
                    value_level1 = hierarchy_aux_dict[value_level2]
                    index_value_level1 = unique_cats_level1.index(value_level1)
                    w_array_new[predicted_l2, index_value_level1] = low_weight_new #establishing a strong relationship between these categories. low penalization
                    w_array_new[index_value_level1, predicted_l2] = low_weight_new
                    for real_1 in range(categories_level1):
                        if real_1 != index_value_level1: #establishing a weak relationship between these categories. high penalization
                            w_array_new[predicted_l2, real_1] = high_weight_new
                            w_array_new[real_1, predicted_l2] = high_weight_new
                elif predicted < categories_level1 and real < categories_level1:  # at least both categories are from level 1
                    w_array_new[predicted, real] = medium_weight_new
                elif predicted > categories_level1 and real > categories_level1:  # at least both categories are from level 2
                    w_array_new[predicted, real] = medium_weight_new
            #else: do nothing.

    # custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_dummy)
    # custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_basic)
    custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_new)
    custom_loss.__name__ = 'w_categorical_crossentropy'

    ################################    LSTM Model:

    dropout = 0.4
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(timesteps, (categories_level1 + categories_level2))))
    model.add(LSTM(units, input_shape=(timesteps, (categories_level1 + categories_level2)), dropout=dropout,
                   return_sequences=False))  # units-> random number. trial and error methodology.
    model.add(Dense((categories_level1 + categories_level2), activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.compile(loss=custom_loss, optimizer='adam', metrics=['acc'])
    print(model.summary())
    # history = LossHistory()
    print("model fit")
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, verbose=0)
    print("model evaluate")
    loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy train: %f' % (accuracy_train * 100))
    loss_test, accuracy_test = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy validation: %f' % (accuracy_test * 100))
    # print(history.losses)
    # print(len(history.losses))

    ################################## test on a new sample. prediction on a SINGLE new data sequence. experiment
    sequence_test_level1 = [140, 140, 140, 140, 140, 859, 653, 1490, 1490, 653]
    sequence_test_level2 = [1519, 1519, 384, 384, 1519, 1473, 312, 145, 798, 92]
    encoded_test_seq_input = []
    for predicted in range(len(sequence_test_level1)):
        value1 = sequence_test_level1[predicted]
        value2 = sequence_test_level2[predicted]
        encoded_test_seq_input.append((one_hot_encode(value1, categories_level1, unique_cats_level1) +
                                       one_hot_encode(value2, categories_level2, unique_cats_level2)))

    if (predicted + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
        for real in range(timesteps - (predicted + 1)):
            encoded_test_seq_input.append(([-1 for _ in range(categories_level1)] +
                                           [-1 for _ in range(categories_level2)]))

    X_test_sample = [encoded_test_seq_input]
    X_test_sample = array(X_test_sample)
    # print(X_test_sample.shape)

    y_test_real_level1 = 653
    y_test_real_level2 = 312

    y_predicted = model.predict(X_test_sample)
    encoded_test_l1 = []
    encoded_test_l2 = []
    for position_seq in X_test_sample[0]:
        encoded_complete_value = position_seq
        encoded_value_level1 = encoded_complete_value[:categories_level1]
        encoded_value_level2 = encoded_complete_value[categories_level1:]
        encoded_test_l1.append(encoded_value_level1.tolist())
        encoded_test_l2.append(encoded_value_level2.tolist())

    decoded_value_level1 = one_hot_decode(encoded_test_l1, unique_cats_level1)
    decoded_value_level2 = one_hot_decode(encoded_test_l2, unique_cats_level2)
    print("Sequence level1: %s" % decoded_value_level1)
    print("Sequence level2: %s" % decoded_value_level2)

    print('Expected level1: %s' % y_test_real_level1)
    print('Expected level2: %s' % y_test_real_level2)

    predicted_l1 = y_predicted[0][:categories_level1]
    predicted_l2 = y_predicted[0][categories_level1:]

    print('Predicted level1: %s' % [predicted_l1], unique_cats_level1[0])
    print('Predicted level2: %s' % [predicted_l2], unique_cats_level2[0])

    print('Predicted decoded level1: %s' % one_hot_decode([predicted_l1], unique_cats_level1)[0])
    print('Predicted decoded level2: %s' % one_hot_decode([predicted_l2], unique_cats_level2)[0])

    ##################################

lstm_model_categorical_data_concat()

'''
##### STANDARD LSTM MODEL. FIRST MODEL:
def lstm_model_categorical_data_first():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')
    unique_cats_level1 = list(df_visitors_no_split['level1'].unique())
    unique_cats_level1.sort()
    print(unique_cats_level1)
    print("Distinct nr of categories level1: %s" % (len(unique_cats_level1)))
    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    #############
    train_pct_index = int(0.40 * len(df_train))
    df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    test_pct_index = int(0.20 * len(df_validation))
    df_validation, df_discard_test = df_validation[:test_pct_index], df_validation[test_pct_index:]
    # ----
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

    ############

    # one hot encode sequence
    def one_hot_encode(sequence, n_features, length):
        encoding = []
        i = 0
        for value in sequence:
            vector = [0 for _ in range(n_features)]
            vector[unique_cats_level1.index(value)] = 1
            encoding.append(vector)
            i += 1
        if i < length:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
            for j in range(length - i):
                encoding.append([-1 for _ in range(n_features)])
        return encoding

    # one hot encode sequence
    def one_hot_encode_target(value, n_features):
        vector = [0 for _ in range(n_features)]
        vector[unique_cats_level1.index(value)] = 1
        return vector

    # decode a one hot encoded sequence
    def one_hot_decode(encoded_seq):
        decoded_seq = []
        # returns index of highest probability value in array.
        for vector in encoded_seq:
            highest_value_index = np.argmax(vector)
            if vector[0] != -1:
                decoded_seq.append(unique_cats_level1[highest_value_index])
        return decoded_seq

    def one_hot_decode_target(encoded_prediction):
        value = unique_cats_level1[encoded_prediction.index(1)]
        return value

    X_train = df_train['sequence_cats_level1']
    y_train = df_train['next_cat_level1']
    X_val = df_validation['sequence_cats_level1']
    y_val = df_validation['next_cat_level1']

    categories_level1 = len(unique_cats_level1)
    max_len = 0
    units = 25
    samples = X_train.shape[0]

    for seq in X_train:
        if len(seq) > max_len:
            max_len = len(seq)

    samples_test = X_val.shape[0]
    for seq in X_val:
        if len(seq) > max_len:
            max_len = len(seq)

    print("max len:")
    print(max_len)

    input_encoded_train = []
    output_encoded_train = []
    for i in range(len(X_train)):
        input_encoded_seq_train = one_hot_encode(X_train[i], categories_level1, max_len)
        input_encoded_train.append(input_encoded_seq_train)
        output_encoded_train.append(one_hot_encode_target(y_train[i], categories_level1))

    input_encoded_test = []
    output_encoded_test = []
    for j in range(len(X_val)):
        input_encoded_seq_test = one_hot_encode(X_val[j], categories_level1, max_len)
        input_encoded_test.append(input_encoded_seq_test)
        output_encoded_test.append(one_hot_encode_target(y_val[j], categories_level1))

    X_train = array(input_encoded_train)
    print("X_train pré: " + str(X_train.shape))
    X_train = X_train.reshape(samples, max_len, categories_level1)
    print("X_train pós: " + str(X_train.shape))
    y_train = array(output_encoded_train)
    print("y_train shape:" + str(y_train.shape))

    X_val = array(input_encoded_test)
    X_val = X_val.reshape(samples_test, max_len, categories_level1)
    y_val = array(output_encoded_test)

    dropout = 0.4
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(max_len, categories_level1)))
    model.add(LSTM(units, input_shape=(max_len, categories_level1), dropout=dropout,
                   return_sequences=False))  # units-> random number. trial and error methodology.
    model.add(Dense(categories_level1, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, verbose=0)
    # evaluate the model
    loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy train: %f' % (accuracy_train * 100))
    loss_test, accuracy_test = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy test: %f' % (accuracy_test * 100))

    # prediction on a SINGLE new data sequence. experiment
    sequence_test = [140, 140, 140, 140, 140, 859, 653, 1490, 1490, 653]
    X_test_new = []
    encoded_seq_level1 = one_hot_encode(sequence_test, categories_level1, max_len)
    X_test_new.append(encoded_seq_level1)
    X_test_new = array(X_test_new)
    print(X_test_new.shape)
    y_test_new = 1698
    y_predicted = model.predict(X_test_new)
    print("*****************")
    # print(X_test_new)

    for value in X_test_new:
        print(value)
        print("+++++++++++++++")
        print(one_hot_decode(value))
    # print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])

    print(y_predicted)

# print('Sequence: %s' % [one_hot_decode(X_test_new[0][i]) for i in range(len(X_test_new))])
# print('Expected: %s' % y_test_new)
# print('Predicted: %s' % one_hot_decode(y_predicted))


# lstm_model_categorical_data_first()
'''