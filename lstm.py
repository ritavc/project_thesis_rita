import math
from keras.layers import Dense, Activation
from sklearn.model_selection import train_test_split
import os
import sys
from tensorflow import keras
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Masking
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from sklearn.metrics import mean_squared_error
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import ast
import numpy
from numpy import array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


####### timeseries prediction (forecasting. regression...). attempt failed.
def former_lstm():
    df_train = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_test.csv')
    #############
    train_pct_index = int(0.90 * len(df_train))
    df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    test_pct_index = int(0.30 * len(df_test))
    df_test, df_discard_test = df_test[:test_pct_index], df_test[test_pct_index:]
    # ----
    # maxlen = 429  # from dataset_processing.py
    df_train['cats_seq'] = [ast.literal_eval(cat_list_string) for cat_list_string in df_train['cats_seq']]
    # Iterate over the sequence of column names
    maxlen = 8
    print("before:")
    print(df_train.shape)
    print(df_train.head())
    for index, row in df_train.iterrows():
        if len(row['cats_seq']) > 8:
            df_train.drop(index, inplace=True)

    print("after:")
    print(df_train.shape)
    print(df_train.head())

    padded_train = pad_sequences(df_train['cats_seq'], maxlen=maxlen, padding='post', value=-1, dtype='float32')
    df_test['cats_seq'] = [ast.literal_eval(cat_list_string) for cat_list_string in df_test['cats_seq']]
    padded_test = pad_sequences(df_test['cats_seq'], maxlen=maxlen, padding='post', value=-1, dtype='float32')

    X_train = padded_train  # masked_input
    y_train = df_train['next_cat']
    X_test = padded_test
    y_test = df_test['next_cat']

    ## LSTM input shape: [samples, timesteps, features]
    n_samples = X_train.shape[0]
    timesteps = X_train.shape[1]
    features = 1
    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(timesteps, features)))
    model.add(LSTM(4, return_sequences=False))
    model.add(Dense(1, activation='sigmoid'))

    print('Training...')
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    model.fit(X_train, y_train, epochs=100, verbose=2)
    print('\n')

    # make predictions
    trainPredict = model.predict(X_train)
    testPredict = model.predict(X_test)

    scores_train = model.evaluate(X_train, y_train, batch_size=1, verbose=0)
    print('Train accuracy:', scores_train[1])
    scores_test = model.evaluate(X_test, y_test, batch_size=1, verbose=0)
    print('Test accuracy:', scores_test[1])


def lstm_model_categorical_data():
    df_visitors_no_split = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_sessions_1min.csv')
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv')
    unique_cats = list(df_visitors_no_split['value'].unique())
    unique_cats.sort()
    print(unique_cats)
    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_train.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_test.csv')
    #############

    train_pct_index = int(0.20 * len(df_train))
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
        if i < length:
            for j in range(length - i):
                encoding.append([-1 for _ in range(n_features)])
        return encoding  # [encoding]#array(encoding)

    # one hot encode sequence
    def one_hot_encode_target(value, n_features):
        vector = [0 for _ in range(n_features)]
        vector[unique_cats.index(value)] = 1
        return vector  # array(vector)

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
    maxlen = 0
    units = 25
    samples = X_train.shape[0]

    for seq in X_train:
        if len(seq) > maxlen:
            maxlen = len(seq)

    input_encoded = []
    output_encoded = []
    for i in range(len(X_train)):
        input_encoded_seq = one_hot_encode(X_train[i], categories, maxlen)
        input_encoded.append(input_encoded_seq)
        output_encoded.append(one_hot_encode_target(y_train[i], categories))
    #print(input_encoded)
    #print(output_encoded)
    X_train = array(input_encoded)
    X_train = X_train.reshape(samples, maxlen, categories)
    #print(X_train.shape)
    y_train = array(output_encoded)
    #print(y_train.shape)


    model = Sequential()
    model.add(Masking(mask_value=-1, input_shape=(maxlen, categories)))
    model.add(LSTM(units, input_shape=(maxlen, categories),
                   return_sequences=False))  # units-> random number. trial and error methodology.
    model.add(Dense(categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    print(model.summary())

    # fit the model
    model.fit(X_train, y_train, epochs=50, verbose=0)
    # evaluate the model
    loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
    print('Accuracy: %f' % (accuracy * 100))

    # prediction on new data
    sequence_test = [1002, 1002, 229, 134, 1028, 121, 1103, 872]
    X_test_new = []
    input_encoded_seq_ = one_hot_encode(sequence_test, categories, maxlen)
    X_test_new.append(input_encoded_seq_)
    X_test_new = array(X_test_new)
    print(X_test_new)
    print(X_test_new.shape)
    y_test_new = 1278
    y_predicted = model.predict(X_test_new)

    print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])
    print('Expected: %s' % y_test_new)
    print('Predicted: %s' % one_hot_decode(y_predicted))

lstm_model_categorical_data()
