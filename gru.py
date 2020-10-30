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
import tensorflow as tf
from functools import partial
import itertools
from sklearn import metrics
import scikitplot as skplt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
stderr = sys.stderr
sys.stderr = open(os.devnull, 'w')
sys.stderr = stderr


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


def gru_model_categorical_data_concat():
    unique_cats_level2 = list(itertools.chain.from_iterable(general_tree_category[1].values()))
    unique_cats_level2.sort()
    print("Distinct nr of categories level2: %s" % (len(unique_cats_level2)))
    df_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_val_set.csv')
    # df_train = df_train.loc[(df_train.visitorid == 404403)]
    # df_train = df_train.iloc[15:65]

    # '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set_temp.csv')
    df_validation = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_set.csv')
    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_test_set.csv')
    ############# for testing purposes (less time consuming using a smaller dataset):
    # train_pct_index = int(0.7 * len(df_train))
    # df_train, df_discard_train = df_train[:train_pct_index], df_train[train_pct_index:]
    # val_pct_index = int(0.7 * len(df_validation))
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

    X_train_level2 = df_train['sequence_cats_level2']
    y_train_level2 = df_train['next_cat_level2']
    X_val_level2 = df_validation['sequence_cats_level2']
    y_val_level2 = df_validation['next_cat_level2']
    X_test_level2 = df_test['sequence_cats_level2']
    y_test_level2 = df_test['next_cat_level2']

    # categories_level1 = len(unique_cats_level1)
    categories_level2 = len(unique_cats_level2)
    events_unique = 3
    timesteps = 5
    samples_train = X_train_level2.shape[0]
    samples_val = X_val_level2.shape[0]
    samples_test = X_test_level2.shape[0]
    '''for current_sequence in X_train_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)

    samples_val = X_val_level2.shape[0]
    for current_sequence in X_val_level2:
        if len(current_sequence) > timesteps:
            timesteps = len(current_sequence)'''

    print("TIMESTEPS: " + str(timesteps))
    print("SAMPLES train: " + str(samples_train))
    print("SAMPLES val: " + str(samples_val))
    print("SAMPLES test: " + str(samples_test))

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
    X_train = X_train.reshape(samples_train, timesteps, (categories_level2 + categories_level2))
    y_train = array(output_encoded_train)
    X_val = array(input_encoded_val)
    X_val = X_val.reshape(samples_val, timesteps, (categories_level2 + categories_level2))
    y_val = array(output_encoded_val)

    ################################ Evaluation metrics customization

    def categorical_accuracy_l2(y_true, y_pred):
        return K.cast(K.equal(K.argmax(y_true[:, categories_level2:], axis=-1),
                              K.argmax(y_pred[:, categories_level2:], axis=-1)),
                      K.floatx())

    def recall_at_k_l2(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true[:, categories_level2:] * y_pred[:, categories_level2:], 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true[:, categories_level2:], 0, 1)))
        return true_positives / (possible_positives + K.epsilon())

    # def mrr_at_k(y_true, y_pred):
    #   return tf.metrics.recall_at_k(y_true[:, categories_level2:], y_pred[:, categories_level2:], 5)

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
        # matrix_ones = np.ones((categories_level2, categories_level2))
        # matrix_ones = tf.convert_to_tensor(matrix_ones, np.float32)
        print(matrix)

        def loss(y_true, y_pred):
            print(y_true[:, categories_level2:])
            print(y_pred[:, categories_level2:])

            l2_y_true = K.argmax(y_true[:, categories_level2:],
                                 axis=1)  # gets the INDEX of the max value category in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample
            print(l2_y_true)
            l2_y_pred = K.argmax(y_pred[:, categories_level2:],
                                 axis=1)  # gets the INDEX of the maximum prob category l2 predicted, in the 1-hot encoded array of levels2 categories. for each y_pred dataset sample
            print(l2_y_pred)

            gather_mat = tf.gather_nd(matrix, tf.stack((l2_y_true, l2_y_pred), -1))
            print(
                gather_mat)  # what this does is to look for the value is the matrix corresponding to the relationship between l2_y_true and l2_y_pred. matrix[l2_y_true, l2_y_pred]
            # K.equal(tf.gather_nd(matrix, [0, 1]), K.constant(1)) teste básico. tudo ok.

            cond = K.equal(gather_mat, K.constant(1))
            # if there is a relationship (value in matrix of 1) -> cond is true.

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
        # model.add(Masking(mask_value=-1, input_shape=(timesteps, (categories_level2 + categories_level2))))
        model.add(GRU(units, input_shape=(timesteps, (categories_level2 + categories_level2)), dropout=dropout,
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
                          metrics=[categorical_accuracy_l2, recall_at_k_l2])

        # tensorboard = TensorBoard(log_dir='logs/{}'.format(time()))
        print(model.summary())
        print("model fit")
        # history = model.fit(X_t, y_t, epochs=epochs, verbose=0)
        history = model.fit(X_t, y_t, validation_data=(X_v, y_v), epochs=epochs, verbose=1)
        print("model evaluate")
        print(model.metrics_names)
        loss_train, acc_train, recall_at_5_train = model.evaluate(X_t, y_t, verbose=1)
        # print('loss_train: %f' % (loss_train * 100))
        print('acc_train: %f' % (acc_train * 100))
        print('recall_at_5_train: %f' % (recall_at_5_train * 100))

        loss_val, acc_val, recall_at_5_val = model.evaluate(X_v, y_v, verbose=1)
        # print('loss_val: %f' % (loss_val * 100))
        print('acc_val: %f' % (acc_val * 100))
        print('recall_at_5_val: %f' % (recall_at_5_val * 100))

        # plt.plot(history.history['categorical_accuracy_l2']) #history verificar se aqui nao é devolvido validation acc tbm.
        # plt.show()

        ################################## experiments - using a test set
        X_test_l2 = df_test  # df_train #df_train.loc[(df_train.visitorid == 404403)]
        X_test = X_test_l2['sequence_cats_level2']
        y_test = X_test_l2['next_cat_level2']
        testing_samples_sequences = X_test  # X_train_level2
        testing_samples_next_category = y_test  # y_train_level2
        testing_samples_next_category_predicted = []
        print([testing_samples_next_category])
        print(testing_samples_next_category.shape)
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

            decoded_sequence = one_hot_decode(encoded_test_l2, unique_cats_level2)

            print("----")
            print("Sequence level2: %s" % decoded_sequence)
            print('Expected level2: %s' % y_test_real_level2)

            pred_l2 = y_predicted[0][categories_level2:]
            pred_family_l2 = y_predicted[0][:categories_level2]
            # print("START")
            # print(one_hot_decode([pred_family_l2], unique_cats_level2)[0])
            # print('Predicted level2: %s' % [pred_l2], unique_cats_level2[0])
            # print("END")
            decoded_value_l2 = one_hot_decode([pred_l2], unique_cats_level2)[0]
            print('Predicted decoded level2: %s' % decoded_value_l2)
            testing_samples_next_category_predicted.append(decoded_value_l2)
            if one_hot_decode([pred_l2], unique_cats_level2)[0] == y_test_real_level2:
                gotten_right += 1
            else:
                gotten_wrong += 1
                print("*--------> wrong prediction!")
            print(i)
        print("nr of right predictions = %s" % gotten_right)
        print("nr of wrong predictions = %s" % gotten_wrong)
        print(testing_samples_next_category_predicted)
        print(len(testing_samples_next_category_predicted))

        confusion_matrix = metrics.confusion_matrix(testing_samples_next_category,
                                                    testing_samples_next_category_predicted)
        print(confusion_matrix)
        print(metrics.classification_report(testing_samples_next_category, testing_samples_next_category_predicted,
                                            digits=3))
        skplt.metrics.plot_confusion_matrix(
            testing_samples_next_category,
            testing_samples_next_category_predicted,
            figsize=(12, 12))
        # plt.show()

    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    build_model(X_train, y_train, X_val, y_val, 40, 0.2, 20, 'adagrad', 'standard',
                'standard categorical crossentropy loss')
    print("++++++++++++++++++++++++++++++++++++++++++++++++")
    # build_model(X_train, y_train, X_val, y_val, 40, 0.2, 150, 'adagrad', 'custom', 'custom loss')

    ##################################


gru_model_categorical_data_concat()


def gru_model_categorical_data():
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
    model.add(GRU(units, input_shape=(max_len, categories),
                  return_sequences=False))  # units-> random number. trial and error methodology.
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
    # print(X_test_new)
    # print(X_test_new.shape)
    y_test_new = 1278
    y_predicted = model.predict(X_test_new)

    print('Sequence: %s' % [one_hot_decode(x) for x in X_test_new])
    print('Expected: %s' % y_test_new)
    print('Predicted: %s' % one_hot_decode(y_predicted))


gru_model_categorical_data()
