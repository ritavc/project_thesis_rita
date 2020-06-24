import ast
import os
import sys

import numpy as np
import pandas as pd
from keras.layers import LSTM, Input, Masking, Dense, Concatenate
from keras.models import Sequential
from numpy import array
from keras.models import Model
import tensorflow as tf
import keras as keras
import keras.backend as K


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

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

    """def loss(target, output, from_logits=False, axis=-1):
        Categorical crossentropy between an output tensor and a target tensor.
            # Arguments
                output: A tensor resulting from a softmax
                    (unless `from_logits` is True, in which
                    case `output` is expected to be the logits).
                target: A tensor of the same shape as `output`.
                from_logits: Boolean, whether `output` is the
                    result of a softmax, or is a tensor of logits.
            # Returns
                Output tensor.

        # Note: tf.nn.softmax_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # scale preds so that the class probas of each sample sum to 1
            output /= tf.reduce_sum(output,
                                    reduction_indices=len(output.get_shape()) - 1,
                                    keep_dims=True)
            # manual computation of crossentropy
            epsilon = _to_tensor(_EPSILON, output.dtype.base_dtype)
            output = tf.clip_by_value(output, epsilon, 1. - epsilon)
            return - tf.reduce_sum(target * tf.log(output),
                                   reduction_indices=len(output.get_shape()) - 1)
        else:
            return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                           logits=output)
        """

    # custom loss function
    def loss(y_true, y_pred):
        # y_true_array = y_true.eval(session=tf.compat.v1.Session())
        #cannot use eval inside loss function...
        #loss_cat_crossentropy = tf.keras.losses.CategoricalCrossentropy()
        #l = loss_cat_crossentropy(y_true, y_pred)
        #d = K.print_tensor(l)
        y_true = K.print_tensor(y_pred)
        y_pred = K.print_tensor(y_pred)
        return K.categorical_crossentropy(y_true, y_pred)
       # return l


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

    print("max len:")
    print(timesteps)  # same for level1 and level2 obviously.
    print("samples in training set:")
    print(samples)
    print("samples in validation set:")
    print(samples_val)

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
            for j in range(timesteps - (v + 1)):
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
            for j in range(timesteps - (v + 1)):
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

    dropout = 0.4
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(timesteps, (categories_level1+categories_level2))))
    model.add(LSTM(units, input_shape=(timesteps, (categories_level1+categories_level2)), dropout=dropout,
                   return_sequences=False))  # units-> random number. trial and error methodology.
    model.add(Dense((categories_level1+categories_level2), activation='softmax'))
    #model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    model.compile(loss=loss, optimizer='adam', metrics=['acc'])
    print(model.summary())
    history = LossHistory()
    # fit the model
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=3, verbose=0, callbacks=[history])
    # evaluate the model
    loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy train: %f' % (accuracy_train * 100))
    loss_test, accuracy_test = model.evaluate(X_val, y_val, verbose=0)
    print('Accuracy validation: %f' % (accuracy_test * 100))
    print(history.losses)
    print(len(history.losses))


    ######## test on a new sample. prediction on a SINGLE new data sequence. experiment
    sequence_test_level1 = [140, 140, 140, 140, 140, 859, 653, 1490, 1490, 653]
    sequence_test_level2 = [1519, 1519, 384, 384, 1519, 1473, 312, 145, 798, 92]
    encoded_test_seq_input = []
    for i in range(len(sequence_test_level1)):
        value1 = sequence_test_level1[i]
        value2 = sequence_test_level2[i]
        encoded_test_seq_input.append((one_hot_encode(value1, categories_level1, unique_cats_level1) +
                                       one_hot_encode(value2, categories_level2, unique_cats_level2)))

    if (i + 1) < timesteps:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
        for j in range(timesteps - (i + 1)):
            encoded_test_seq_input.append(([-1 for _ in range(categories_level1)] +
                                           [-1 for _ in range(categories_level2)]))

    X_test_sample = [encoded_test_seq_input]
    X_test_sample = array(X_test_sample)
    #print(X_test_sample.shape)

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

    print('Predicted level1: %s' % one_hot_decode([predicted_l1], unique_cats_level1)[0])
    print('Predicted level2: %s' % one_hot_decode([predicted_l2], unique_cats_level2)[0])

lstm_model_categorical_data_concat()


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
    #print(X_test_new)

    for value in X_test_new:
        print(value)
        print("+++++++++++++++")
        print(one_hot_decode(value))
    # print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])

    print(y_predicted)

#print('Sequence: %s' % [one_hot_decode(X_test_new[0][i]) for i in range(len(X_test_new))])
    #print('Expected: %s' % y_test_new)
    #print('Predicted: %s' % one_hot_decode(y_predicted))


#lstm_model_categorical_data_first()
