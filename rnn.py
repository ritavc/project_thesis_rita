import ast
import os
import sys
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.layers import SimpleRNN
from keras.layers import Masking
from keras.layers import Embedding
from keras.models import Sequential
from numpy import array
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dropout
from keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


def rnn_model_categorical_data():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_sessions_1min.csv')
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv')
    unique_cats = list(df_visitors_no_split['value'].unique())
    unique_cats.sort()
    # print(unique_cats)
    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_train.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_test.csv')
    #############
    #train_pct_index = int(0.40 * len(df_train))
    #df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    #test_pct_index = int(0.20 * len(df_test))
    #df_test, df_discard_test = df_test[:test_pct_index], df_test[test_pct_index:]
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
        if i < length:  # -1 as a value chosen to ignore (mask) positions in the sequence with no values
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

    def fit_model(X_train, y_train, X_test, y_test, recurrent):

        # batch size: nr of sequences that are trained together. Very big batch size may not fit into memory and takes a long time to be trained.The disadvantage it is less accurate
        # stateful: If True, the last state for each sample at index i in a batch will be used as initial state for the sample of index i in the following batch.
        model = Sequential()
        model.add(Masking(mask_value=-1, input_shape=(max_len, categories)))
        if recurrent == 0:
            model.add(SimpleRNN(units, input_shape=(max_len, categories), activation='tanh', use_bias=True,
                                return_sequences=False))  # units-> random number. trial and error methodology.
            print("SimpleRNN")
        elif recurrent == 1:
            model.add(LSTM(units, input_shape=(max_len, categories), activation='tanh', use_bias=True,
                           return_sequences=False))
            print("LSTM")

        else:
            model.add(GRU(units, input_shape=(max_len, categories), activation='tanh', use_bias=True,
                          return_sequences=False))
            print("GRU")
        model.add(Dropout(rate=0.2))
        model.add(Dense(categories, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
        print(model.summary())

        # fit the model
        history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, verbose=0)
        # evaluate the model
        loss_train, accuracy_train = model.evaluate(X_train, y_train, verbose=0)
        print('--- Accuracy train: %f' % (accuracy_train * 100))
        loss_test, accuracy_test = model.evaluate(X_test, y_test, verbose=0)
        print('--- Accuracy test: %f' % (accuracy_test * 100))

        # prediction on SINGLE new data sequence

        sequence_test = [1002, 1002, 229, 134, 1028, 121, 1103, 872]
        X_test_new = []
        input_encoded_seq_ = one_hot_encode(sequence_test, categories, max_len)
        X_test_new.append(input_encoded_seq_)
        X_test_new = array(X_test_new)
        # print(X_test_new)
        # print(X_test_new.shape)
        y_test_new = 1278
        y_predicted = model.predict(X_test_new)

        print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])
        print('Expected: %s' % y_test_new)
        print('Predicted: %s' % one_hot_decode(y_predicted))

        # PLOT
        plt.ylim(0.0, 1.0)
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        # plt.title('dropout=' + str(dropout), pad=-80)
        plt.legend(['train', 'val'], loc='upper left')
        # plt.show()

    X_train = df_train['cats_seq']
    y_train = df_train['next_cat']
    X_test = df_test['cats_seq']
    y_test = df_test['next_cat']

    categories = 328
    max_len = 0
    units = 50
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

    momentums = ['sgd', 'rmsprop', 'adagrad', 'adam']
    dropouts = [0, 0.2, 0.4, 0.6]
    models = [0, 1, 2] # 0- RNN, 1- LSTM, 2- GRU
    for i in range(len(models)):
        # determine the plot number
        plot_no = 220 + (i + 1)
        plt.subplot(plot_no)
        # fit model and plot learning curves for an optimizer
        fit_model(X_train, y_train, X_test, y_test, models[i])
    # show learning curves
    plt.show()




rnn_model_categorical_data()
