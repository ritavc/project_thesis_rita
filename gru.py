import ast
import os
import sys

import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import GRU
from keras.layers import Dropout
from keras.layers import Masking
from keras.models import Sequential
from numpy import array
import keras.backend as K
from itertools import product
from functools import partial


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

def dummy_example_weigths_matrix():
    categories_level1 = 2
    categories_level2 = 5
    unique_cats_level1 = ['A', 'B']
    unique_cats_level2 = ['C', 'D', 'E', 'F', 'G']
    hierarchy_aux_dict = {'C': 'A', 'D': 'A', 'F': 'B', 'E': 'B', 'G': 'B'}

    ## C) Weigths matrix real costumization:
    w_array_new = np.ones(((categories_level1 + categories_level2), (categories_level1 + categories_level2)))
    high_relation = 0.8
    medium_relation = 0.5
    no_relation = 0
    for predicted in range((categories_level1 + categories_level2)):
        for real in range(predicted, (categories_level1 + categories_level2)):
            if predicted != real:
                if predicted < categories_level1 and real >= categories_level1:  # when predicted and real categories are from different levels. establish here a relationship parent-child.
                    predicted_l1 = predicted
                    real_l2 = real
                    value_level1_predicted = unique_cats_level1[predicted_l1]
                    value_level2_real = unique_cats_level2[real_l2 - categories_level1]
                    value_level1_real = hierarchy_aux_dict[value_level2_real]

                    if value_level1_predicted == value_level1_real:
                        w_array_new[
                            predicted_l1, real_l2] = high_relation  # establishing a parent-child relationship between these categories.
                        w_array_new[real_l2, predicted_l1] = high_relation
                    else:
                        w_array_new[predicted_l1, real_l2] = no_relation
                        w_array_new[real_l2, predicted_l1] = no_relation

                elif predicted < categories_level1 and real < categories_level1:  # both categories are different and from level 1
                    w_array_new[predicted, real] = no_relation
                    w_array_new[real, predicted] = no_relation

                elif predicted >= categories_level1 and real >= categories_level1:  # both categories are different and from level 2
                    predicted_l2 = predicted
                    real_l2 = real
                    value_level2_predicted = unique_cats_level2[predicted_l2 - categories_level1]
                    value_level1_predicted = hierarchy_aux_dict[value_level2_predicted]
                    value_level2_real = unique_cats_level2[real_l2 - categories_level1]
                    value_level1_real = hierarchy_aux_dict[value_level2_real]
                    if value_level1_predicted == value_level1_real:
                        w_array_new[
                            predicted_l2, real_l2] = medium_relation  # establishing a "brother" relationship between these categories.
                        w_array_new[real_l2, predicted_l2] = medium_relation
                    else:
                        w_array_new[predicted_l2, real_l2] = no_relation
                        w_array_new[real_l2, predicted_l2] = no_relation
            # else: do nothing.
    print(w_array_new)

#dummy_example_weigths_matrix()

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

def gru_model_categorical_data_concat():
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

    print("TIMESTEPS: "+str(timesteps))
    print("SAMPLES: "+str(samples))

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
    '''## A) dummy weigths matrix construction:
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
    '''
    ## C) Weigths matrix real costumization:
    w_array_new = np.ones(((categories_level1 + categories_level2), (categories_level1 + categories_level2)))
    high_relation = 0.1
    medium_relation = 0.5
    no_relation = 1 #high penalization?
    for predicted in range((categories_level1 + categories_level2)):
        for real in range(predicted, (categories_level1 + categories_level2)):
            if predicted != real:
                if predicted < categories_level1 and real >= categories_level1:  # when predicted and real categories are from different levels. establish here a relationship parent-child.
                    predicted_l1 = predicted
                    real_l2 = real
                    value_level1_predicted = unique_cats_level1[predicted_l1]
                    value_level2_real = unique_cats_level2[real_l2 - categories_level1]
                    value_level1_real = hierarchy_aux_dict[value_level2_real]

                    if value_level1_predicted == value_level1_real:
                        w_array_new[
                            predicted_l1, real_l2] = high_relation  # establishing a parent-child relationship between these categories.
                        w_array_new[real_l2, predicted_l1] = high_relation
                    else:
                        w_array_new[predicted_l1, real_l2] = no_relation
                        w_array_new[real_l2, predicted_l1] = no_relation

                elif predicted < categories_level1 and real < categories_level1:  # both categories are different and from level 1
                    w_array_new[predicted, real] = no_relation
                    w_array_new[real, predicted] = no_relation

                elif predicted >= categories_level1 and real >= categories_level1:  # both categories are different and from level 2
                    predicted_l2 = predicted
                    real_l2 = real
                    value_level2_predicted = unique_cats_level2[predicted_l2 - categories_level1]
                    value_level1_predicted = hierarchy_aux_dict[value_level2_predicted]
                    value_level2_real = unique_cats_level2[real_l2 - categories_level1]
                    value_level1_real = hierarchy_aux_dict[value_level2_real]
                    if value_level1_predicted == value_level1_real:
                        w_array_new[
                            predicted_l2, real_l2] = medium_relation  # establishing a "brother" relationship between these categories.
                        w_array_new[real_l2, predicted_l2] = medium_relation
                    else:
                        w_array_new[predicted_l2, real_l2] = no_relation
                        w_array_new[real_l2, predicted_l2] = no_relation
            # else: do nothing.
    print("weights matrix constructed.")
    # custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_dummy)
    # custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_basic)
    custom_loss = partial(w_categorical_crossentropy_levels, weights=w_array_new)
    custom_loss.__name__ = 'w_categorical_crossentropy'

    ################################    LSTM Model:

    dropout = 0.4
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(timesteps, (categories_level1 + categories_level2))))
    model.add(GRU(units, input_shape=(timesteps, (categories_level1 + categories_level2)), dropout=dropout,
                   return_sequences=False))  # units-> random number. trial and error methodology.
    #model.add(Dense((categories_level1 + categories_level2), activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
    sequence_test_level1 = [653, 1490, 1490, 653]
    sequence_test_level2 = [312, 145, 798, 92]
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

gru_model_categorical_data_concat()





def gru_model_categorical_data():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_sessions_1min.csv')
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv')
    unique_cats = list(df_visitors_no_split['value'].unique())
    unique_cats.sort()
    #print(unique_cats)
    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_train.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_test.csv')
    #############
    train_pct_index = int(0.40 * len(df_train))
    df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    test_pct_index = int(0.20 * len(df_test))
    df_test, df_discard_test = df_test[:test_pct_index], df_test[test_pct_index:]
    # ----
    df_train['cats_seq'] = [ast.literal_eval(cat_list_string) for cat_list_string in df_train['cats_seq']]
    df_test['cats_seq'] = [ast.literal_eval(cat_list_string) for cat_list_string in df_test['cats_seq']]
    ############

    # one hot encode sequence
    def one_hot_encode(sequence, n_features, length):
        encoding = []
        i = 0
        for value in sequence:
            vector = [0 for _ in range(n_features)]
            vector[unique_cats.index(value)] = 1
            encoding.append(vector)
            i += 1
        if i < length: #-1 as a value chosen to ignore (mask) positions in the sequence with no values
            for j in range(length - i):
                encoding.append([-1 for _ in range(n_features)])
        return encoding

    # one hot encode sequence
    def one_hot_encode_target(value, n_features):
        vector = [0 for _ in range(n_features)]
        vector[unique_cats.index(value)] = 1
        return vector

    # decode a one hot encoded string
    def one_hot_decode(encoded_seq):
        decoded_seq = []
        # returns index of highest probability value in array.
        for vector in encoded_seq:
            highest_value_index = np.argmax(vector)
            if vector[0] != -1:
                decoded_seq.append(unique_cats[highest_value_index])
        return decoded_seq

    def one_hot_decode_target(encoded_prediction):
        value = unique_cats[encoded_prediction.index(1)]
        return value

    X_train = df_train['cats_seq']
    y_train = df_train['next_cat']
    X_test = df_test['cats_seq']
    y_test = df_test['next_cat']

    categories = 328
    max_len = 0
    units = 25
    samples = X_train.shape[0]

    for seq in X_train:
        if len(seq) > max_len:
            max_len = len(seq)

    samples_test = X_test.shape[0]
    for seq in X_test:
        if len(seq) > max_len:
            max_len = len(seq)

    print("max len:")
    print(max_len)

    input_encoded_train = []
    output_encoded_train = []
    for i in range(len(X_train)):
        input_encoded_seq_train = one_hot_encode(X_train[i], categories, max_len)
        input_encoded_train.append(input_encoded_seq_train)
        output_encoded_train.append(one_hot_encode_target(y_train[i], categories))

    input_encoded_test = []
    output_encoded_test = []
    for j in range(len(X_test)):
        input_encoded_seq_test = one_hot_encode(X_test[j], categories, max_len)
        input_encoded_test.append(input_encoded_seq_test)
        output_encoded_test.append(one_hot_encode_target(y_test[j], categories))

    X_train = array(input_encoded_train)
    X_train = X_train.reshape(samples, max_len, categories)
    y_train = array(output_encoded_train)

    X_test = array(input_encoded_test)
    X_test = X_test.reshape(samples_test, max_len, categories)
    y_test = array(output_encoded_test)

    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(max_len, categories)))
    model.add(GRU(units, input_shape=(max_len, categories), return_sequences=False))  # units-> random number. trial and error methodology.
    model.add(Dropout(rate=0.2))
    model.add(Dense(categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)
    # evaluate the model
    loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy train: %f' % (accuracy_train * 100))
    loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
    print('Accuracy test: %f' % (accuracy_test * 100))


    # prediction on SINGLE new data sequence
    sequence_test = [1002, 1002, 229, 134, 1028, 121, 1103, 872]
    X_test_new = []
    input_encoded_seq_ = one_hot_encode(sequence_test, categories, max_len)
    X_test_new.append(input_encoded_seq_)
    X_test_new = array(X_test_new)
    #print(X_test_new)
    #print(X_test_new.shape)
    y_test_new = 1278
    y_predicted = model.predict(X_test_new)

    print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])
    print('Expected: %s' % y_test_new)
    print('Predicted: %s' % one_hot_decode(y_predicted))

gru_model_categorical_data()
