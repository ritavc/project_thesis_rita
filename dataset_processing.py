import numpy as np
import pandas as pd
import seaborn as sns
import collections
import datetime
import matplotlib.pyplot as plt
import random
from datetime import timedelta
import ast

pd.options.display.max_columns
import dataset_analysis

df_cat = pd.read_csv(
    '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')  # , converters={'parentid': lambda x: str(x)})
df_cat = df_cat.fillna(-1)


# df_cat = df_cat.astype(int)


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

    flatten_level1 = list_root_categories
    flatten_level2 = []
    flatten_level3 = []
    flatten_level4 = []
    flatten_level5 = []
    flatten_level6 = []

    aux_dict_2 = {}
    aux_dict_3 = {}
    aux_dict_4 = {}
    aux_dict_5 = {}
    aux_dict_6 = {}

    for cat in list_root_categories:  ###level 2
        categories_next_level = df_cat.loc[(df_cat.parentid == int(cat))]
        l = list(categories_next_level.categoryid)
        l.sort()
        tree_dict[cat] = l
        flatten_level2.extend(l)
        # aux_dict_2 = tree_dict
        for elem in l:
            aux_dict_2[elem] = cat

    for key, value in tree_dict.items():  ###level 3
        parent_level1 = key
        for i in range(len(value)):
            categories_next_level = df_cat.loc[(df_cat.parentid.astype(int) == int(value[i]))]
            l = list(categories_next_level.categoryid)
            parent_level2 = value[i]
            if len(l) > 0:
                aux_dict = dict()
                aux_dict[parent_level2] = l
                tree_dict[parent_level1][i] = aux_dict
                flatten_level3.extend(l)
                # aux_dict_3[parent_level2] = l
                for elem in l:
                    aux_dict_3[elem] = parent_level2

    for key_1, value_1 in tree_dict.items():  ###level 4
        parent_level1 = key_1
        for j in range(len(value_1)):
            if isinstance(value_1[j], dict):
                for key_2, value_2 in value_1[j].items():
                    parent_level2 = key_2
                    for i in range(len(value_2)):
                        parent_level3 = value_2[i]
                        categories_next_level = df_cat.loc[(df_cat.parentid == parent_level3)]
                        l = list(categories_next_level.categoryid)
                        if len(l) > 0:
                            aux_dict = dict()
                            aux_dict[parent_level3] = l
                            tree_dict[parent_level1][j][parent_level2][i] = aux_dict
                            flatten_level4.extend(l)
                            # aux_dict_4[parent_level3] = l
                            for elem in l:
                                aux_dict_4[elem] = parent_level3

    for key_1, value_1 in tree_dict.items():  ###level 5
        parent_level1 = key_1
        for w in range(len(value_1)):
            if isinstance(value_1[w], dict):
                for key_2, value_2 in value_1[w].items():
                    parent_level2 = key_2
                    for j in range(len(value_2)):
                        if isinstance(value_2[j], dict):
                            for key_3, value_3 in value_2[j].items():
                                parent_level3 = key_3
                                for i in range(len(value_3)):
                                    parent_level4 = value_3[i]
                                    categories_next_level = df_cat.loc[(df_cat.parentid == parent_level4)]
                                    l = list(categories_next_level.categoryid)
                                    if len(l) > 0:
                                        aux_dict = dict()
                                        aux_dict[parent_level4] = l
                                        tree_dict[parent_level1][w][parent_level2][j][parent_level3][i] = aux_dict
                                        flatten_level5.extend(l)
                                        # aux_dict_5[parent_level4] = l
                                        for elem in l:
                                            aux_dict_5[elem] = parent_level4

    for key_1, value_1 in tree_dict.items():  ###level 6
        parent_level1 = key_1
        for z in range(len(value_1)):
            if isinstance(value_1[z], dict):
                for key_2, value_2 in value_1[z].items():
                    parent_level2 = key_2
                    for w in range(len(value_2)):
                        if isinstance(value_2[w], dict):
                            for key_3, value_3 in value_2[w].items():
                                parent_level3 = key_3
                                for j in range(len(value_3)):
                                    if isinstance(value_3[j], dict):
                                        for key_4, value_4 in value_3[j].items():
                                            parent_level4 = key_4
                                            for i in range(len(value_4)):
                                                parent_level5 = value_4[i]
                                                categories_next_level = df_cat.loc[(df_cat.parentid == parent_level5)]
                                                l = list(categories_next_level.categoryid)
                                                if len(l) > 0:
                                                    aux_dict = dict()
                                                    aux_dict[parent_level5] = l
                                                    tree_dict[parent_level1][z][parent_level2][w][parent_level3][j][
                                                        parent_level4][i] = aux_dict
                                                    flatten_level6.extend(l)
                                                    for elem in l:
                                                        aux_dict_6[elem] = parent_level5

    return tree_dict, flatten_level1, flatten_level2, flatten_level3, flatten_level4, flatten_level5, flatten_level6, aux_dict_2, aux_dict_3, aux_dict_4, aux_dict_5, aux_dict_6


# category_tree()

def assign_corresponding_tree_cat_levels():
    df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
    cats_level6 = []
    cats_level5 = []
    cats_level4 = []
    cats_level3 = []
    cats_level2 = []
    cats_level1 = []

    result = category_tree()
    flatten_level1, flatten_level2, flatten_level3, flatten_level4, flatten_level5, flatten_level6 = result[1], result[
        2], result[3], result[4], result[5], result[6]
    aux_dict_2, aux_dict_3, aux_dict_4, aux_dict_5, aux_dict_6 = result[7], result[8], result[9], result[10], result[11]

    for index, row in df.iterrows():
        current_category = int(row['value'])
        category_level1 = -1
        category_level2 = -1
        category_level3 = -1
        category_level4 = -1
        category_level5 = -1
        category_level6 = -1

        if current_category in flatten_level1:
            category_level1 = current_category
        elif current_category in flatten_level2:
            category_level2 = current_category
            category_level1 = aux_dict_2[category_level2]
        elif current_category in flatten_level3:
            category_level3 = current_category
            category_level2 = aux_dict_3[category_level3]
            category_level1 = aux_dict_2[category_level2]
        elif current_category in flatten_level4:
            category_level4 = current_category
            category_level3 = aux_dict_4[category_level4]
            category_level2 = aux_dict_3[category_level3]
            category_level1 = aux_dict_2[category_level2]
        elif current_category in flatten_level5:
            category_level5 = current_category
            category_level4 = aux_dict_5[category_level5]
            category_level3 = aux_dict_4[category_level4]
            category_level2 = aux_dict_3[category_level3]
            category_level1 = aux_dict_2[category_level2]
        elif current_category in flatten_level6:
            category_level6 = current_category
            category_level5 = aux_dict_6[category_level6]
            category_level4 = aux_dict_5[category_level5]
            category_level3 = aux_dict_4[category_level4]
            category_level2 = aux_dict_3[category_level3]
            category_level1 = aux_dict_2[category_level2]

        cats_level1.append(category_level1)
        cats_level2.append(category_level2)
        cats_level3.append(category_level3)
        cats_level4.append(category_level4)
        cats_level5.append(category_level5)
        cats_level6.append(category_level6)

    df['level1'], df['level2'], df['level3'], df['level4'], df['level5'], df[
        'level6'] = cats_level1, cats_level2, cats_level3, cats_level4, cats_level5, cats_level6
    unique_level1 = df['level1'].unique()
    print("Distinct nr of categories level1: %s" % (unique_level1.shape,))
    unique_level2 = df['level2'].unique()
    print("Distinct nr of categories level2: %s" % (unique_level2.shape,))
    # df.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv', index=False)


# assign_corresponding_tree_cat_levels()

def timelag_sessions_duplicate_actions():
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv')
    unique_visitors = df_visitors['visitorid'].unique()
    times = []
    for i in df_visitors['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    df_visitors['timestamp'] = times

    indexes_to_remove = []
    for current_visitor in unique_visitors:
        new_set = df_visitors.loc[(df_visitors.visitorid == current_visitor)]
        indexes_new_set = list(new_set.index)
        for i in range(new_set.shape[0] - 1):
            minute_diff = (datetime.datetime.min + (
                    new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i])).time()
            day_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).days
            hour_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).seconds // 3600
            print(new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i])

            if (day_diff == 0) and (hour_diff == 0) and (
                    (minute_diff.minute == 0 and minute_diff.second > 1) or (minute_diff.minute == 1)):
                if new_set['itemid'].iloc[i] == new_set['itemid'].iloc[i + 1] and new_set['event'].iloc[i] == \
                        new_set['event'].iloc[i + 1] and new_set['value'].iloc[i] == new_set['value'].iloc[i + 1]:
                    # delete consecutive entries made by the same user with the same itemID, eventID and same categoryID
                    print("----> 2nd IF")
                    print(new_set.iloc[i])
                    print("---")
                    print(new_set.iloc[i + 1])
                    indexes_to_remove.append(indexes_new_set[i + 1])

    df_visitors.drop(df_visitors.index[indexes_to_remove], inplace=True)
    df_visitors.sort_values(by=['visitorid'], inplace=True)
    print(df_visitors.head(40))
    df_visitors.to_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv',
        index=False)


# timelag_sessions_duplicate_actions()

## division done by spliting a number of users to one side and others to the other side.
# includes different division points.
# includes simple statistical analysis to check how many short/long sequences result in each division performed.
## -> result: csv files 'train_seqs_set' and 'val_seqs_set'
def dataset_split_by_users():
    # df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')
    # df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv')
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv')
    df_visitors = df_visitors[(df_visitors.level2 != int(-1))]
    # times = []
    # for i in df_visitors['timestamp']:
    #   times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    # df_visitors['timestamp'] = times

    unique_visitors = df_visitors[
        'visitorid'].unique()  # dataset_analysis.df_final_creation()  # df_visitors['visitorid'].unique() -> ou RANDOM aqui.
    print(len(unique_visitors))
    random.shuffle(unique_visitors)
    train_val_division_index = int(0.60 * len(unique_visitors))
    unique_train, unique_val_temp = unique_visitors[:train_val_division_index], unique_visitors[
                                                                                train_val_division_index:]
    print(len(unique_train))
    print(len(unique_val_temp))
    val_test_division_index = int(0.60 * len(unique_val_temp))
    unique_val, unique_test = unique_val_temp[:val_test_division_index], unique_val_temp[val_test_division_index:]

    # test_division_index = int(0.20 * len(unique_test_temp))
    # unique_test, nothing = unique_test_temp[:test_division_index], unique_test_temp[test_division_index:]

    df_train = df_visitors[df_visitors['visitorid'].isin(unique_train)]
    df_val = df_visitors[df_visitors['visitorid'].isin(unique_val)]
    df_test = df_visitors[df_visitors['visitorid'].isin(unique_test)]
    print("Nr of instances in dataset train : %s" % (df_train.shape,))
    print("Nr of instances in dataset validation: %s" % (df_val.shape,))
    print("Nr of instances in dataset test: %s" % (df_test.shape,))

    categories_train = df_train['value'].unique()
    print("Distinct nr of categories in train: %s" % (categories_train.shape,))
    users_train = df_train['visitorid'].unique()
    print("Distinct nr of users in train : %s" % (users_train.shape,))
    categories_val = df_val['value'].unique()
    print("Distinct nr of categories in val: %s" % (categories_val.shape,))
    users_val = df_val['visitorid'].unique()
    print("Distinct nr of users in val: %s" % (users_val.shape,))
    categories_test = df_test['value'].unique()
    print("Distinct nr of categories in test: %s" % (categories_test.shape,))
    users_test = df_test['visitorid'].unique()
    print("Distinct nr of users in test: %s" % (users_test.shape,))

    groups_test = df_test.sort_values(['visitorid'], ascending=True).groupby('visitorid')

    df_train.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_train.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', index=False)
    # print(df_train.head())
    df_val.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_val.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', index=False)
    # print(df_val.head())
    df_test.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_test.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv', index=False)
    # print(df_test.head())


# dataset_split_by_users()

list_keys = list(range(2, 43))
dict_distributions = dict.fromkeys(list_keys, 0)


def inactivity_window_sessions_division_variable(path, new_path, minutes_div):
    print("-> sessions divisions by " + str(minutes_div) + " minutes")
    df = pd.read_csv(path,
                     parse_dates=['timestamp'])
    df['event'] = df['event'].astype('category')
    df['event'] = df['event'].cat.codes
    data_dict = {'visitorid': [], 'session': [], 'sequence_items': [], 'next_itemId': [], 'sequence_events': [],
                 'sequence_cats_level1': [],
                 'next_cat_level1': [],
                 'sequence_cats_level2': [], 'next_cat_level2': []}
    unique_visitors = df['visitorid'].unique()
    maximum_length = 0
    total_events = 0
    for current_visitor in unique_visitors:
        events = []
        new_set = df.loc[(df.visitorid == current_visitor)]
        session_sequence_l1 = []
        session_sequence_l2 = []
        session_sequence_items = []
        session_counter = 0
        for i in range(new_set.shape[0] - 1):
            minute_diff = (datetime.datetime.min + (
                    new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i])).time()
            day_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).days
            hour_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).seconds // 3600
            # print(new_set_train['timestamp'].iloc[i + 1] - new_set_train['timestamp'].iloc[i])

            if (day_diff == 0) and (hour_diff == 0) and (minute_diff.minute <= minutes_div):

                if len(session_sequence_l1) == 0:
                    session_sequence_l1.append(new_set['level1'].iloc[i])
                    session_sequence_l2.append(new_set['level2'].iloc[i])
                    session_sequence_items.append(new_set['itemid'].iloc[i])
                    session_sequence_l1.append(new_set['level1'].iloc[i + 1])
                    session_sequence_l2.append(new_set['level2'].iloc[i + 1])
                    session_sequence_items.append(new_set['itemid'].iloc[i + 1])

                    if new_set['level2'].iloc[i] == -1 or new_set['level2'].iloc[i + 1] == -1:
                        print("*** -1 in " + str(current_visitor))
                else:
                    session_sequence_l1.append(new_set['level1'].iloc[i + 1])
                    session_sequence_l2.append(new_set['level2'].iloc[i + 1])
                    session_sequence_items.append(new_set['itemid'].iloc[i + 1])
                    if new_set['level2'].iloc[i + 1] == -1:
                        print("*** -1 in " + str(current_visitor))

            if (i + 1) == new_set.shape[0] - 1 or (
                    (day_diff != 0) or (hour_diff != 0) or (minute_diff.minute > minutes_div)):

                if len(session_sequence_l1) >= 2:
                    if maximum_length < len(session_sequence_l1):
                        maximum_length = len(session_sequence_l1)
                    dict_distributions[len(session_sequence_l1)] = dict_distributions.get(
                        len(session_sequence_l1)) + 1

                    total_events += len(session_sequence_l1)
                    real_next_cat_level1 = session_sequence_l1.pop()
                    real_next_cat_level2 = session_sequence_l2.pop()
                    real_next_itemid = session_sequence_items.pop()
                    data_dict['next_cat_level1'].append(real_next_cat_level1)
                    data_dict['next_cat_level2'].append(real_next_cat_level2)
                    data_dict['next_itemId'].append(real_next_itemid)
                    data_dict['sequence_cats_level1'].append(session_sequence_l1)
                    data_dict['sequence_cats_level2'].append(session_sequence_l2)
                    data_dict['sequence_items'].append(session_sequence_items)
                    data_dict['sequence_events'].append(events)
                    data_dict['session'].append(session_counter)
                    data_dict['visitorid'].append(current_visitor)

                    session_counter += 1
                session_sequence_l1 = []
                session_sequence_l2 = []
                session_sequence_items = []

    df_new = pd.DataFrame(data_dict)
    print(df_new.head(20))
    print("new number of sessions instances = " + str(df_new.shape[0]))
    print("nr diff visitors = " + str(len(df_new['visitorid'].unique())))
    print("TOTAL EVENTS = " + str(total_events))
    df_new.to_csv(new_path, index=False)


'''print("###")
print("TRAIN")
inactivity_window_sessions_division_variable('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_train_set.csv', 3)
print("VALIDATION")
inactivity_window_sessions_division_variable('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_val_set.csv', 3)
print("TEST")
inactivity_window_sessions_division_variable('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv','/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/variable_test_set.csv', 3)

print(dict_distributions)
plt.bar(dict_distributions.keys(), dict_distributions.values())
plt.show()'''


def fixed_size_sessions_moving_window(path, new_path, window_events, min_div):
    fixed_window_lookback = window_events  # in reality, 1 event lookback. 2 events form a sequence.
    print("-> sessions divisions by " + str(min_div) + " minutes")
    df = pd.read_csv(path,
                     parse_dates=['timestamp'])
    df['event'] = df['event'].astype('category')
    df['event'] = df['event'].cat.codes
    data_dict = {'visitorid': [], 'session': [], 'sequence_items': [], 'next_itemId': [], 'sequence_events': [],
                 'sequence_cats_level1': [],
                 'next_cat_level1': [],
                 'sequence_cats_level2': [], 'next_cat_level2': []}
    unique_visitors = df['visitorid'].unique()
    maximum_length = 0
    one_event_sessions = 0
    for current_visitor in unique_visitors:

        events = []
        new_set = df.loc[(df.visitorid == current_visitor)]
        session_sequence_l1 = []
        session_sequence_l2 = []
        session_sequence_items = []
        for i in range(new_set.shape[0] - 1):
            minute_diff = (datetime.datetime.min + (
                    new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i])).time()
            day_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).days
            hour_diff = (new_set['timestamp'].iloc[i + 1] - new_set['timestamp'].iloc[i]).seconds // 3600
            # print(new_set_train['timestamp'].iloc[i + 1] - new_set_train['timestamp'].iloc[i])
            if current_visitor == 3465:
                print(current_visitor)
                print(minute_diff.minute)
                print(new_set['itemid'].iloc[i])
                print(new_set['itemid'].iloc[i + 1])
            if (day_diff == 0) and (hour_diff == 0) and (minute_diff.minute <= min_div):
                if len(session_sequence_l1) == 0:
                    session_sequence_l1.append(new_set['level1'].iloc[i])
                    session_sequence_l2.append(new_set['level2'].iloc[i])
                    session_sequence_items.append(new_set['itemid'].iloc[i])
                    session_sequence_l1.append(new_set['level1'].iloc[i + 1])
                    session_sequence_l2.append(new_set['level2'].iloc[i + 1])
                    session_sequence_items.append(new_set['itemid'].iloc[i + 1])
                    if new_set['level2'].iloc[i] == -1 or new_set['level2'].iloc[i + 1] == -1:
                        print("*** -1 in " + str(current_visitor))
                else:
                    session_sequence_l1.append(new_set['level1'].iloc[i + 1])
                    session_sequence_l2.append(new_set['level2'].iloc[i + 1])
                    session_sequence_items.append(new_set['itemid'].iloc[i + 1])
                    if new_set['level2'].iloc[i + 1] == -1:
                        print("*** -1 in " + str(current_visitor))
                if current_visitor == 3465:
                    print("first:")
                    print(session_sequence_items)
            if (i + 1) == new_set.shape[0] - 1 or (
                    (day_diff != 0) or (hour_diff != 0) or (minute_diff.minute > min_div)):
                if current_visitor == 3465:
                    print("division to-do")
                if len(session_sequence_l1) >= 2:
                    if maximum_length < len(session_sequence_l1):
                        maximum_length = len(session_sequence_l1)
                    dict_distributions[len(session_sequence_l1)] = dict_distributions.get(
                        len(session_sequence_l1)) + 1

                    for k in range(0, len(session_sequence_l1)):
                        session_sequence_l1_break = session_sequence_l1[k:k + fixed_window_lookback]

                        if len(session_sequence_l1_break) < fixed_window_lookback:
                            break
                        session_sequence_l2_break = session_sequence_l2[k:k + fixed_window_lookback]
                        session_sequence_items_break = session_sequence_items[k:k + fixed_window_lookback]
                        data_dict['visitorid'].append(current_visitor)
                        data_dict['session'].append(k)

                        real_next_cat_level1 = session_sequence_l1_break.pop()
                        real_next_cat_level2 = session_sequence_l2_break.pop()
                        real_next_itemid = session_sequence_items_break.pop()
                        data_dict['next_cat_level1'].append(real_next_cat_level1)
                        data_dict['next_cat_level2'].append(real_next_cat_level2)
                        data_dict['next_itemId'].append(real_next_itemid)
                        data_dict['sequence_cats_level1'].append(session_sequence_l1_break)
                        data_dict['sequence_cats_level2'].append(session_sequence_l2_break)
                        data_dict['sequence_items'].append(session_sequence_items_break)
                        data_dict['sequence_events'].append(events)
                if current_visitor == 3465:
                    print(data_dict['sequence_items'])
                    print(data_dict['next_itemId'])
                session_sequence_l1 = []
                session_sequence_l2 = []
                session_sequence_items = []

    df_new = pd.DataFrame(data_dict)
    print(df_new.head(20))
    print("new number of sessions instances = " + str(df_new.shape[0]))
    print("nr diff visitors = " + str(len(df_new['visitorid'].unique())))
    print("1-event sessions = " + str(one_event_sessions))
    df_new.to_csv(new_path, index=False)


'''print("###")
print("TRAIN set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_train_set.csv', 2, 3)
print("VALIDATION set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set.csv', 2, 3)
print("TEST set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv',
                                  '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_test_set.csv', 2, 3)'''
'''
print("###")
print("TRAIN set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_2.csv', 2, 3)
print("VALIDATION set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_2.csv', 2, 3)
print("TEST set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv','/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_2.csv', 2, 3)
print("###")
print("TRAIN set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_3.csv', 3, 3)
print("VALIDATION set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_3.csv', 3, 3)
print("TEST set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv','/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_3.csv', 3, 3)
print("###")
print("TRAIN set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_4.csv', 4, 3)
print("VALIDATION set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_4.csv', 4, 3)
print("TEST set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv','/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_4.csv', 4, 3)
print("###")
print("TRAIN set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_5.csv', 5, 3)
print("VALIDATION set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv', '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_5.csv', 5, 3)
print("TEST set: ")
fixed_size_sessions_moving_window('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv','/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_5.csv', 5, 3)

'''


def unregular_test_set(df_from, df_to):
    df = pd.read_csv(df_from)
    df['sequence_cats_level2'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                                  df['sequence_cats_level2']]
    df['sequence_items'] = [ast.literal_eval(cat_list_string) for cat_list_string in
                            df['sequence_items']]
    data_train = []
    for index, row in df.iterrows():
        data_row = []
        if int(row['sequence_cats_level2'][0]) != int(row['next_cat_level2']) and int(row['sequence_items'][0]) != int(row['next_itemId']):
            data_row.append([int(row['sequence_cats_level2'][0])])
            data_row.append(int(row['next_cat_level2']))
            data_row.append([int(row['sequence_items'][0])])
            data_row.append(int(row['next_itemId']))
        if len(data_row) > 0:
            data_train.append(data_row)

    df_new = pd.DataFrame(data_train,
                          columns=['sequence_cats_level2', 'next_cat_level2', 'sequence_items', 'next_itemId'])

    print(df_new.shape)
    df_new.to_csv(df_to, index=False)


unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_train_set.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/diff_train_set.csv')
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_val_set.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/diff_val_set.csv')
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_test_set.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/diff_test_set.csv')
print("###")
print("TRAIN set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_2.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_train_set_2.csv')
print("VALIDATION set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_2.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_val_set_2.csv')
print("TEST set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_2.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_test_set_2.csv')
print("###")
print("TRAIN set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_3.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_train_set_3.csv')
print("VALIDATION set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_3.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_val_set_3.csv')
print("TEST set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_3.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_test_set_3.csv')
print("###")
print("TRAIN set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_4.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_train_set_4.csv')
print("VALIDATION set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_4.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_val_set_4.csv')
print("TEST set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_4.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_test_set_4.csv')
print("###")
print("TRAIN set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_train_set_5.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_train_set_5.csv')
print("VALIDATION set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_val_set_5.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_val_set_5.csv')
print("TEST set: ")
unregular_test_set('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/fixed_test_set_5.csv',
                   '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/n_grams/diff_test_set_5.csv')
