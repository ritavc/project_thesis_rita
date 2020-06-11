import numpy as np
import pandas as pd
import seaborn as sns
import collections
import datetime
pd.options.display.max_columns
df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')#, converters={'parentid': lambda x: str(x)})
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
    #print(root_cats)
    return tree_dict

def category_tree():
    tree_dict = empty_initial_categories_dict() ###level 1
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

    for cat in list_root_categories: ###level 2
        categories_next_level = df_cat.loc[(df_cat.parentid == int(cat))]
        l = list(categories_next_level.categoryid)
        l.sort()
        tree_dict[cat] = l
        flatten_level2.extend(l)
        #aux_dict_2 = tree_dict
        for elem in l:
            aux_dict_2[elem] = cat

    for key, value in tree_dict.items(): ###level 3
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
                #aux_dict_3[parent_level2] = l
                for elem in l:
                    aux_dict_3[elem] = parent_level2

    for key_1, value_1 in tree_dict.items(): ###level 4
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
                            #aux_dict_4[parent_level3] = l
                            for elem in l:
                                aux_dict_4[elem] = parent_level3

    for key_1, value_1 in tree_dict.items(): ###level 5
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
                                        #aux_dict_5[parent_level4] = l
                                        for elem in l:
                                            aux_dict_5[elem] = parent_level4

    for key_1, value_1 in tree_dict.items(): ###level 6
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
                                                    tree_dict[parent_level1][z][parent_level2][w][parent_level3][j][parent_level4][i] = aux_dict
                                                    flatten_level6.extend(l)
                                                    for elem in l:
                                                        aux_dict_6[elem] = parent_level5

    return tree_dict, flatten_level1, flatten_level2, flatten_level3, flatten_level4, flatten_level5, flatten_level6, aux_dict_2, aux_dict_3, aux_dict_4, aux_dict_5, aux_dict_6
#category_tree()

def assign_corresponding_tree_cat_levels():
    df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
    cats_level6 = []
    cats_level5 = []
    cats_level4 = []
    cats_level3 = []
    cats_level2 = []
    cats_level1 = []

    result = category_tree()
    flatten_level1, flatten_level2, flatten_level3, flatten_level4, flatten_level5, flatten_level6 = result[1], result[2], result[3], result[4], result[5], result[6]
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

    df['level1'], df['level2'], df['level3'], df['level4'], df['level5'], df['level6'] = cats_level1, cats_level2, cats_level3, cats_level4, cats_level5, cats_level6
    df.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv', index=False)

#assign_corresponding_tree_cat_levels()

def timelag_sessions_division():
    df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv')
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
            minute_diff = (datetime.datetime.min + (new_set['timestamp'].iloc[i+1] - new_set['timestamp'].iloc[i])).time()
            day_diff = (new_set['timestamp'].iloc[i+1] - new_set['timestamp'].iloc[i]).days

            if (day_diff == 0) and ((minute_diff.minute == 0 and minute_diff.second > 1) or (minute_diff.minute == 1)):
                if new_set['itemid'].iloc[i] == new_set['itemid'].iloc[i+1] and new_set['event'].iloc[i] == new_set['event'].iloc[i+1] and new_set['value'].iloc[i] == new_set['value'].iloc[i+1]:
                    #delete consecutive entries made by the same user with the same eventID and same categoryID
                    indexes_to_remove.append(indexes_new_set[i+1])

    df_visitors.drop(df_visitors.index[indexes_to_remove], inplace=True)
    print(df_visitors.head(40))
    df_visitors.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_timelag_separation.csv', index=False)

#timelag_sessions_division()

def group_sessions():
    df_final = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_timelag_separation.csv')

    print("Nr of instances in joined dataset: %s " % (df_final.shape,))

    #print("maximum cat: %s " % np.amax(df_final.value.astype(int)))
    #print((df_final['value'].unique()).shape)

    groups = df_final.sort_values(['visitorid'], ascending=True).groupby('visitorid')
    groups_size = groups.size()
    print(groups_size)
    bigger = 0
    smaller = 0
    list_visitorId_consider = []
    for i, row in groups_size.iteritems():
        if int(row) > 3:
            bigger += 1
            list_visitorId_consider.append(i)
        else:
            smaller += 1
        #print(i, row)
    print("bigger: "+str(bigger))
    print("smaller: "+str(smaller))
    axs_x = ['shorter sequences w/ less or equal to 3 events', 'seq w/ more than 3 events']
    axs_y = [smaller, bigger]
    sns.barplot(axs_x, axs_y)#, ax=axs[0])
    #plt.show()
    print(list_visitorId_consider)
    df_final = df_final[df_final['visitorid'].isin(list_visitorId_consider)]
    #print(df_final.head(30))
    df_final = df_final[['visitorid', 'timestamp', 'event', 'itemid', 'transactionid', 'property', 'value', 'level1', 'level2', 'level3', 'level4', 'level5', 'level6']]
    df_final.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)

    #attention: deleting here 3 entries in the dataset that only have level1 category and not its corresponding level2 category.
    #in order to fit the purpose of predicting both category levels of the event this decision was made here.
    df_final = df_final[(df_final.level2 != int(-1))]
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv', index=False)

#group_sessions()

## simple statistical analysis of the newly created dataset 'joined_events_items_longer_seqs'
def experiments_groupings_cats():
    df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv')
    groups = df_visitors.groupby('visitorid')
    print("Nr of instances in dataset: %s" % (df_visitors.shape,))
    #print(groups.size().value_counts().head(60))
    print("maximum: %s "%np.amax(df_visitors.value))
    print("minimum: %s " % np.amin(df_visitors.value))
    categories = df_visitors['value'].unique()
    print("Distinct nr of categories: %s" % (categories.shape,))
    users = df_visitors['visitorid'].unique()
    print("Distinct nr of users: %s" % (users.shape,))


## experiments to split the dataset into train and test sets. division done by spliting a number of users to one side and others to the other side.
#includes different division points.
#includes simple statistical analysis to check how many short/long sequences result in each division performed.
## -> result: csv files 'joined_train' and 'joined_test'
def dataset_split_by_users():
    df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_longer_seqs.csv')

    df_visitors.sort_values(by=['visitorid'], inplace=True)
    #2890 users to one side, 1239 users to the other side. train with 2890 users.
    # X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    # y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    unique_visitors = df_visitors['visitorid'].unique()
    train_pct_index = int(0.60 * len(unique_visitors))
    unique_train, unique_test = unique_visitors[:train_pct_index], unique_visitors[train_pct_index:]
    df_train = df_visitors[df_visitors['visitorid'].isin(unique_train)]
    df_test = df_visitors[df_visitors['visitorid'].isin(unique_test)]
    print("Nr of instances in dataset train : %s" % (df_train.shape,))

    print("Nr of instances in dataset test: %s" % (df_test.shape,))

    categories_train = df_train['value'].unique()
    print("Distinct nr of categories: %s" % (categories_train.shape,))
    users_train = df_train['visitorid'].unique()
    print("Distinct nr of users train : %s" % (users_train.shape,))
    categories_test = df_test['value'].unique()
    print("Distinct nr of categories: %s" % (categories_test.shape,))
    users_test = df_test['visitorid'].unique()
    print("Distinct nr of users test : %s" % (users_test.shape,))

    groups_train = df_train.sort_values(['visitorid'], ascending=True).groupby('visitorid')
    groups_size_train = groups_train.size()
    print("groups train: ")
    print(groups_size_train)
    bigger_train = 0
    smaller_train = 0
    list_long_train_unique = []
    for i, row in groups_size_train.iteritems():
        if int(row) > 3:
            bigger_train += 1
            list_long_train_unique.append(i)
        else:
            smaller_train += 1
        # print(i, row)

    print("number of smaller seqs in train : {}".format(smaller_train))
    print("number of bigger seqs in train : {}".format(bigger_train))

    groups_test = df_test.sort_values(['visitorid'], ascending=True).groupby('visitorid')
    groups_size_test = groups_test.size()
    print("groups test: ")
    print(groups_size_test)
    bigger_test = 0
    smaller_test = 0
    list_long_test = []
    list_small_test = []
    for i, row in groups_size_test.iteritems():
        if int(row) > 3:
            bigger_test += 1
            list_long_test.append(i)
        else:
            smaller_test += 1
            list_small_test.append(i)

        # print(i, row)
    print("number of smaller seqs in test : {}".format(smaller_test))
    print("number of bigger seqs in test : {}".format(bigger_test))

    df_train.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_train.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_seqs_set.csv', index=False)

    df_test.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_test.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_seqs_set.csv', index=False)

dataset_split_by_users()






## function to shift the dataset in order to be in the form: per row, X sequence of categories (taken from each event) and y next real category
## -> result: csv files 'shifted_train' and 'shifted_test'
## includes, a minmaxscaling of the categories as well, and this experiment is saved in csv files 'shifted_train_cat_scaling' and 'shifted_test_cat_scaling'
def prepare_dataset_seqs_target():
    df_train = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_train_level1.csv')
    #df_train['value'] = scaler.fit_transform(df_train[['value']])

    data_train = {'visitorid': [], 'cats_seq': [], 'next_cat': []}
    #for index, row in df_train.iterrows():
    unique_visitors_train = df_train['visitorid'].unique()
    max_len_cats_sequence = 0
    for current_visitor_train in unique_visitors_train:
        X = []
        new_set = df_train.loc[(df_train.visitorid == current_visitor_train)]
        data_train['visitorid'].append(current_visitor_train)
        for cat in new_set['value']:
            #X.append(float(cat))
            X.append(int(cat))
        real_next_cat = X.pop()
        data_train['next_cat'].append(real_next_cat)
        if len(X) > max_len_cats_sequence:
            max_len_cats_sequence = len(X)
        data_train['cats_seq'].append(X)
    #print(data_train)
    df_train_new = pd.DataFrame(data_train)
    print(df_train_new.head(20))
    #df_train_new.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_train_cat_scaling.csv', index=False)
    df_train_new.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_train_cats_level1.csv',
                        index=False)

    df_test = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_test_level1.csv')
    #df_test['value'] = scaler.fit_transform(df_test[['value']])

    data_test = {'visitorid': [], 'cats_seq': [], 'next_cat': []}
    unique_visitors_test = df_test['visitorid'].unique()
    for current_visitor_test in unique_visitors_test:
        X = []
        new_set = df_test.loc[(df_test.visitorid == current_visitor_test)]
        data_test['visitorid'].append(current_visitor_test)
        for cat in new_set['value']:
            #X.append(float(cat))
            X.append(int(cat))
        real_next_cat = X.pop()
        data_test['next_cat'].append(real_next_cat)
        if len(X) > max_len_cats_sequence:
            max_len_cats_sequence = len(X)
        data_test['cats_seq'].append(X)
    #print(data_test)
    df_test_new = pd.DataFrame(data_test)
    print(df_test_new.head(20))

    print("max len seqs = "+str(max_len_cats_sequence))
    #df_test_new.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/shifted_test_cat_scaling.csv', index=False)
    df_test_new.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/shifted_test_cats_level1.csv',
                       index=False)

#prepare_dataset_seqs_target_cats_level1()