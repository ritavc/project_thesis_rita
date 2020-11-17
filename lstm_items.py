import ast
import os
import sys
from functools import partial
import matplotlib.pyplot as plt
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences

import numpy as np
import pandas as pd
from keras.layers import LSTM, Input, Masking, Dense, Concatenate, Embedding, Reshape, Flatten
from keras.models import Sequential
from numpy import array
from keras.preprocessing import text, sequence
from keras.models import Model
import tensorflow as tf
from keras.utils import to_categorical
import keras as keras
import keras.backend as K
from itertools import product
import itertools
from sklearn import metrics
from keras.callbacks import TensorBoard
from time import time
import seaborn as sn
import scikitplot as skplt
from sklearn.metrics import recall_score

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


def dict_items_cats_relations():
    df = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv')
    df = df[(df.level2 != int(-1))]
    items_cats_dict = {}

    for index, row in df.iterrows():
        if int(row['itemid']) not in items_cats_dict:
            items_cats_dict[int(row['itemid'])] = int(row['level2'])
    cats_items_sorted_items_values = {k: v for k, v in sorted(items_cats_dict.items(), key=lambda item: item[1])}
    #print(cats_items_sorted_items_values)
    return cats_items_sorted_items_values


# one hot encode value of sequence
def one_hot_encode(value, n_features, unique_elems):
    vector = [0 for _ in range(n_features)]
    vector[unique_elems.index(value)] = 1
    return vector


# decode a one hot encoded sequence
def one_hot_decode(encoded_seq, unique_elems):
    decoded_seq = []
    # returns index of highest probability value in array.
    for vector in encoded_seq:
        # print(vector)
        highest_value_index = np.argmax(vector)
        if vector[0] != -1:
            decoded_seq.append(unique_elems[highest_value_index])
    return decoded_seq


def lstm_item_prediction_custom_loss():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv')
    unique_cats_level1 = list(df_visitors_no_split['level1'].unique())
    unique_cats_level1.sort()
    # print(unique_cats_level1)
    # print("Distinct nr of categories level1: %s" % (len(unique_cats_level1)))
    unique_cats_level2 = list(df_visitors_no_split['level2'].unique())
    unique_cats_level2.sort()
    unique_itemsIds = list(dict_items_cats_relations().keys())  # list(df_visitors_no_split['itemid'].unique())
    # unique_itemsIds.sort()
    print("---")
    print(unique_itemsIds)
    print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))
    print("Distinct nr of items IDs: %s" % (len(unique_itemsIds)))

    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_train_set.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_val_set.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_test_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    # train_pct_index = int(0.0005 * len(df_train))
    # df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    # val_pct_index = int(0.0005 * len(df_validation))
    # df_validation, df_discard_val = df_validation[:val_pct_index], df_validation[val_pct_index:]
    # test_pct_index = int(0.1 * len(df_test))
    # df_test, df_discard_test = df_test[:test_pct_index], df_test[test_pct_index:]
    ############

    indexes_train = df_train.index.values
    indexes_validation = df_validation.index.values

    df_train['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level1']]
    df_validation['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level1']]
    df_test['sequence_cats_level1'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                       df_test['sequence_cats_level1']]
    df_train['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                        df_train['sequence_cats_level2']]
    df_validation['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                             df_validation['sequence_cats_level2']]
    df_test['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                       df_test['sequence_cats_level2']]
    df_train['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                   df_train['sequence_events']]
    df_validation['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                        df_validation['sequence_events']]
    df_test['sequence_events'] = [ast.literal_eval(event_list_string) for event_list_string in
                                  df_test['sequence_events']]
    df_train['sequence_items'] = [ast.literal_eval(event_list_string) for event_list_string in
                                  df_train['sequence_items']]
    df_validation['sequence_items'] = [ast.literal_eval(event_list_string) for event_list_string in
                                       df_validation['sequence_items']]
    df_test['sequence_items'] = [ast.literal_eval(event_list_string) for event_list_string in
                                 df_test['sequence_items']]

    X_train_items = df_train['sequence_items']
    y_train_items = df_train['next_itemId']
    X_train_level2 = df_train['sequence_cats_level2']
    y_train_level2 = df_train['next_cat_level2']
    X_val_items = df_validation['sequence_items']
    y_val_items = df_validation['next_itemId']
    X_test_items = df_test['sequence_items']
    y_test_items = df_test['next_itemId']

    categories_level2 = len(unique_cats_level2)
    items_ids_total = len(unique_itemsIds)
    timesteps = 0
    samples_train = X_train_items.shape[0]
    samples_val = X_val_items.shape[0]

    '''for current_sequence in X_train_items:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_items.shape[0]
    for current_sequence in X_val_items:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    print("TIMESTEPS: " + str(timesteps))
    print("SAMPLES train: " + str(samples_train))
    print("SAMPLES val: " + str(samples_val))

    input_encoded_train = []
    output_encoded_train = []
    for x in indexes_train:
        one_sequence_encoding_input = []
        curr_sequence = X_train_items[x]
        for v in range(len(curr_sequence)):
            value = X_train_items[x][v]
            one_sequence_encoding_input.append(
                unique_itemsIds.index(value))  # one_hot_encode(value2, categories_level2, unique_cats_level2))

        next_item = y_train_items[x]

        output_encoded_train.append(one_hot_encode(next_item, items_ids_total, unique_itemsIds))
        # unique_itemsIds.index(next_item))  # one_hot_encode(next_item, items_ids_total, unique_itemsIds))
        input_encoded_train.append(one_sequence_encoding_input)

    input_encoded_val = []
    output_encoded_val = []
    for z in indexes_validation:
        one_sequence_encoding_input = []
        curr_sequence = X_val_items[z]
        for v in range(len(curr_sequence)):
            value = X_val_items[z][v]
            one_sequence_encoding_input.append(unique_itemsIds.index(value))

        next_item = y_val_items[z]
        output_encoded_val.append(one_hot_encode(next_item, items_ids_total, unique_itemsIds))
        # unique_itemsIds.index(next_item))  #
        input_encoded_val.append(one_sequence_encoding_input)

    X_array_train = []
    for val in X_train_items.values:
        X_array_train.append(val)
    X_array_train = array(input_encoded_train)
    y_train = array(output_encoded_train)

    X_array_val = []
    for val in X_val_items.values:
        X_array_val.append(val)
    X_array_val = array(input_encoded_val)
    y_val = array(output_encoded_val)'''

    ################################        Loss Function Customization - Function Construction:
    relations_items_cats_aux_dict = dict_items_cats_relations()

    def matrix_relations_construction():
        mat = np.zeros((items_ids_total, items_ids_total))
        for pred in range(items_ids_total):
            for true in range(items_ids_total):
                parent_pred = relations_items_cats_aux_dict[unique_itemsIds[pred]]
                parent_true = relations_items_cats_aux_dict[unique_itemsIds[true]]
                if parent_pred == parent_true:
                    mat[true, pred] = 1
                    mat[pred, true] = 1
        return mat

    def loss_function(matrix):
        matrix = tf.constant(matrix, np.float32)
        # matrix_ones = np.ones((categories_level2, categories_level2))
        # matrix_ones = tf.convert_to_tensor(matrix_ones, np.float32)
        print(matrix)

        def loss(y_true, y_pred):
            print(y_true)
            print(y_pred)

            index_y_true = K.argmax(y_true,
                                    axis=1)  # gets the INDEX of the max value category in the 1-hot encoded array of items Ids. for each y_pred dataset sample
            print(index_y_true)
            index_y_pred = K.argmax(y_pred,
                                    axis=1)  # gets the INDEX of the maximum prob category itemId predicted, in the 1-hot encoded array of items Ids. for each y_pred dataset sample
            print(index_y_pred)

            gather_mat = tf.gather_nd(matrix, tf.stack((index_y_true, index_y_pred), -1))
            print(
                gather_mat)  # looks for the value in the matrix corresponding to the relationship between index_y_true and index_y_pred. matrix[index_y_true, index_y_pred]
            # K.equal(tf.gather_nd(matrix, [0, 1]), K.constant(1)) teste básico. tudo ok.

            cond = K.equal(gather_mat, K.constant(1))
            # if there is a relationship (value in matrix of 1) -> cond is true.

            return K.switch(cond,
                            # if the predicted l2 category is a sibling of the true l2 category -> turn the loss value lower by multiplying it by 0.1
                            lambda: K.categorical_crossentropy(y_true,
                                                               y_pred,
                                                               from_logits=False) * 0.1,
                            lambda: K.categorical_crossentropy(y_true,
                                                               y_pred,
                                                               from_logits=False))

        return loss

    def categorical_crossentropy(y_true, y_pred):
        cross_ent = K.categorical_crossentropy(y_true, y_pred,
                                               from_logits=False)
        return cross_ent

    def loss_function_custom_again(mapping):
        list_items = tf.constant(list(mapping.keys()), np.float32)
        list_corresponding_categories = tf.constant(list(mapping.values()), np.float32)

        def loss(y_true, y_pred):
            print(y_true)
            print(y_pred)

            index_y_true = K.argmax(y_true,
                                    axis=1)  # gets the INDEX of the max value category in the 1-hot encoded array of items Ids. for each y_pred dataset sample
            print(index_y_true)
            index_y_pred = K.argmax(y_pred,
                                    axis=1)  # gets the INDEX of the maximum prob category itemId predicted, in the 1-hot encoded array of items Ids. for each y_pred dataset sample
            print(index_y_pred)

            cross_ent = K.categorical_crossentropy(y_true[:, 2:145], y_pred[:, 2:145], from_logits=False)

            return cross_ent
        return loss

    ######## acc metric
    def categorical_accuracy(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true[:, 2:145], axis=-1),
                              K.argmax(y_pred[:, 2:145], axis=-1)),
                      K.floatx())


    ################################    LSTM Model:
    def build_model(X_t, y_t, X_v, y_v, units, dropout, epochs, opt, loss, title):
        metric_l2 = partial(categorical_accuracy)
        metric_l2.__name__ = 'categorical_accuracyl2'
        print("MODEL " + str(title))
        units = units
        dropout = dropout
        model = Sequential()
        vocab_size = items_ids_total
        embedding_size = min(50, vocab_size // 2 + 1)
        embedding_size = int(embedding_size)
        print(embedding_size)
        model.add(Embedding(vocab_size, embedding_size))
        # model.add(Reshape(target_shape=(embedding_size, timesteps)))
        # model.add(Masking(mask_value=-1, input_shape=(timesteps, items_ids_total)))
        model.add(LSTM(units=embedding_size,
                       return_sequences=False))
        model.add(Dense(vocab_size, activation='softmax'))
        if loss == 'standard':
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=[categorical_accuracy])
        elif loss == 'other':
            model.compile(loss='categorical_crossentropy', optimizer=opt,
                          metrics=['acc'])
        else:
            model.compile(loss=loss_function_custom_again(dict_items_cats_relations()), optimizer=opt, metrics=[categorical_accuracy])
            #model.compile(loss=loss_function(matrix_relations_construction()), optimizer=opt,metrics=['acc'])
        print("model done")
        print(model.summary())
        # history = LossHistory()
        print("model fit")
        history = model.fit(X_t, y_t, epochs=epochs, verbose=2)#validation_data=(X_v, y_v), verbose=0)
        print("model evaluate")
        print(model.metrics_names)

        loss_train, accuracy_train = model.evaluate(X_t, y_t, verbose=0)
        print('Accuracy train: %f' % (accuracy_train * 100))
        '''loss_val, accuracy_val = model.evaluate(X_v, y_v, verbose=0)
        print('Accuracy validation: %f' % (accuracy_val * 100))

        ################################## experiments - using a test set

        X_test_l2 = df_test  # df_train #df_train.loc[(df_train.visitorid == 404403)]
        X_test = X_test_l2['sequence_items']
        y_test = X_test_l2['next_itemId']
        testing_samples_sequences = X_test  # X_train_level2
        testing_samples_next_item = y_test  # y_train_level2
        testing_samples_next_item_predicted = []

        gotten_right = 0
        gotten_wrong = 0
        gotten_right_top_5 = 0
        reciprocal_ranks = 0
        for i in range(len(testing_samples_sequences)):
            sample_sequence = testing_samples_sequences.iloc[i]
            encoded_test_seq_input = []
            for pred in range(len(sample_sequence)):
                item = sample_sequence[pred]
                encoded_test_seq_input.append(unique_itemsIds.index(item))

            if (pred + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
                for true in range(timesteps - (pred + 1)):
                    encoded_test_seq_input.append(([-1 for _ in range(categories_level2)] +
                                                   [-1 for _ in range(categories_level2)]))

            X_test_sample = [encoded_test_seq_input]
            X_test_sample = array(X_test_sample)
            # print(X_test_sample.shape)
            encoded_test = []
            

            y_predicted = model.predict(X_test_sample)
            y_test_real_level2 = testing_samples_next_item.iloc[i]
            pred_item_encoded = y_predicted[0][:]
            # decoded_seq = one_hot_decode(encoded_test, unique_itemsIds)
            value_pred_item = one_hot_decode([pred_item_encoded], unique_itemsIds)[0]
            top_indexes = sorted(range(len(pred_item_encoded)), key=lambda d: pred_item_encoded[d], reverse=True)[:5]
            top_values = [unique_cats_level2[a] for a in top_indexes]
            # print(top_values)
            reciprocal_rank = 0
            if y_test_real_level2 in top_values:
                gotten_right_top_5 += 1
                reciprocal_rank = 1 / (top_values.index(y_test_real_level2) + 1)
            reciprocal_ranks += reciprocal_rank

            print("----")
            print("Sequence items Ids: %s" % unique_itemsIds[X_test_sample[0][0]])
            print('Expected next item Id: %s' % y_test_real_level2)
            print('Predicted decoded next item Id: %s' % value_pred_item)
            testing_samples_next_item_predicted.append(value_pred_item)
            if value_pred_item == y_test_real_level2:
                gotten_right += 1
            else:
                gotten_wrong += 1
                print("*--------> wrong prediction!")
        print("nr of right predictions = %s" % gotten_right)
        print("nr of wrong predictions = %s" % gotten_wrong)
        print("accuracy = %s" % (gotten_right / len(testing_samples_next_item_predicted)))
        print("recall_at_5 = %s" % (gotten_right_top_5 / len(testing_samples_next_item_predicted)))
        print("mrr_at_5 = %s" % (reciprocal_ranks / len(testing_samples_next_item_predicted)))
        '''
        print("finito model")

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
    X = np.array([[398634], [150932], [258819], [390857], [83446], [212564], [271705]])
    y = np.array([398634, 391598, 170021, 150932, 328901, 258819, 238932])
    input_encoded_train = []
    output_encoded_train = []
    i=0
    for x in X:
        one_sequence_encoding_input = []
        curr_sequence = x
        value = X_train_items[0]
        one_sequence_encoding_input.append(
                unique_itemsIds.index(value[0]))  # one_hot_encode(value2, categories_level2, unique_cats_level2))

        next_item = y[i]

        output_encoded_train.append(one_hot_encode(next_item, items_ids_total, unique_itemsIds))
        # unique_itemsIds.index(next_item))  # one_hot_encode(next_item, items_ids_total, unique_itemsIds))
        input_encoded_train.append(one_sequence_encoding_input)
        i += 1

    X_array_train = []
    for val in X_train_items.values:
        X_array_train.append(val)
    X_array_train = array(input_encoded_train)
    y_train = array(output_encoded_train)

    build_model(X_array_train, y_train, 0, 0, items_ids_total / 2, 0, 50, 'adam', 'standard', 'standard categorical crossentropy loss')
    build_model(X_array_train, y_train, 0, 0, items_ids_total / 2, 0, 50, 'adam', 'other', 'custom loss')
    build_model(X_array_train, y_train, 0, 0, items_ids_total / 2, 0, 50, 'adam', 'custom', 'truly custom loss')


lstm_item_prediction_custom_loss()





def bla():
    X = np.array([[3], [6], [5], [200]])
    # y = to_categorical(np.array([5, 9, 7, 4]))
    y = np.array([5, 8, 7, 4])
    model = Sequential()
    embedding_size = min(50, 300 // 2 + 1)
    embedding_size = int(embedding_size)
    print(embedding_size)
    model.add(Embedding(300, embedding_size))
    # model.add(Reshape(target_shape=(embedding_size, timesteps)))
    # model.add(Masking(mask_value=-1, input_shape=(timesteps, items_ids_total)))
    model.add(LSTM(units=embedding_size,
                   return_sequences=False))
    model.add(Dense(300, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',
                  metrics=['acc'])
    model.fit(X, y, epochs=10, verbose=2)
# bla()
