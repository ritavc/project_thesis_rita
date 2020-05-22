import numpy as np
import pandas as pd
import seaborn as sns

df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')

df_cat = df_cat.fillna(-1) ## just to simplify the reading of the csv, since there are categories on the top of the tree that naturally have the parentid field empty.
df_cat['parentid'] = df_cat['parentid'].astype(int)


def nested_search(parentId, dictionary, result):
    #print("parentId : {}".format(parentId))
    #global result
    # result = {'msg': 'none', 'key': -1, 'path': []}
    if dictionary:
        for key, value in dictionary.items():
            result['path'].append(key)
            #print(key, '->', value)
            #print("...at the beginning, path : {}".format(result['path']))
            result['key'] = key
            if parentId in value:
                #print("- if parentId in value")
                #print("valuebla: {}".format(value.index(parentId)))
                result['msg'] = 'another level'
                #result['key'] = key
                result['path'].append((value.index(parentId)))
                #print("-> result: ")
                #print(result)
                return result
            elif key == parentId:
                #print("- elif key == parentId")
                result['msg'] = 'append level'
                #result['key'] = key
                #result['path'].append(key)
                #print("-> result: ")
                #print(result)
                return result
            elif isinstance(value, dict):
                #print("- elif isinstance(value, dict)")
                #result['path'].append(key)
                nested_search(parentId, value, result)
            elif isinstance(value, list):
                #print("- elif isinstance(value, list)")
                if len(value) > 0:
                    #print("if len(value) > 0")
                    for i in range(0, len(value)):
                        #print("Result in list loop: {}".format(result))
                        #print("item : {}".format(value[i]))
                        if result['msg'] != 'none':
                            return result
                        elif value[i] == parentId:
                            #print("- if item == parentId")
                            # print("INDEX = {}".format(i))
                            result['path'].append(i)
                            result['path'].append(parentId)
                            result['msg'] = 'another level'
                            #print("-> result: ")
                            #print(result)
                            return result
                        elif isinstance(value[i], dict):
                            #print("IN LIST - elif isinstance(item, dict)")
                            #print("index = {}".format(i))
                            result['path'].append(i)
                            result['msg'] = 'none'
                            #print("-> result: ")
                            #print(result)
                            nested_search(parentId, value[i], result)
                        elif value[i] != parentId:
                            #print("- elif item != parentId")
                            result['path'].append(i)
                            result['msg'] = 'none'

                        #print("going for another round_IN List")

                        if (len(result['path']) > 0) & (result['msg'] == 'none'):
                            #print("if (len(result['path']) > 0) & (result['msg'] == 'none')")
                            del result['path'][-1] #index elimination
                            #print("-> result: ")
                            #print(result)

                else: # else para o caso de value com lista vazia []
                    #print("else: len(value) <= 0")
                    if len(result['path']) > 0:
                        del result['path'][-1]
                        #print("-> result: ")
                        #print(result)

                #print("out of for loop in isinstance(value, list)")

                if result['msg'] != 'none':
                    #print("if result['msg'] != 'none'")
                    #print("-> result: ")
                    #print(result)
                    return result
                '''else:
                    if len(result['path']) > 0:
                        del result['path'][-1]
                        print("else result['msg'] == 'none'")
                        print("-> result: ")
                        print(result)'''

            #print("going for another round_IN Dictionary")

            if (len(result['path']) > 0) & (result['msg'] == 'none'):
                #print("if (len(result['path']) > 0) & (result['msg'] == 'none')")
                del result['path'][-1]  # key elimination
                #print("-> result: ")
                #print(result)

        #print(result)
        return result
    else:
        #print(result)
        return result

def category_tree():
    i = 0
    tree_dict = dict() # defaultdict(list)
    for index, row in df_cat.iterrows():
        #print("______" + str(i))
        categoryId = int(row['categoryid'])

        if row['parentid'] == -1:
            tree_dict[categoryId] = []
        else:
            #print(tree_dict)
            parentId = int(row['parentid'])
            result_in = {'msg': 'none', 'key': -1, 'path': []}
            result_nested = nested_search(parentId, tree_dict, result_in)
            #print("RESULT FINAL : ")
            #print(result_nested)
            if result_nested['msg'] == 'another level':
                # print("*** 1): if result[0] == 'another level' ***")
                aux_dict = dict()
                aux_dict[parentId] = [categoryId]
                list_path = result_nested['path']
                l = len(list_path)
                if l == 1:
                    tree_dict[list_path[0]] = aux_dict
                elif l == 2:
                    tree_dict[list_path[0]][list_path[1]] = aux_dict
                elif l == 3:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]] = aux_dict
                elif l == 4:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]] = aux_dict
                elif l == 5:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]][list_path[4]] = aux_dict
                elif l == 6:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]][list_path[4]][
                        list_path[5]] = aux_dict

            elif result_nested['msg'] == 'append level':
                # print("*** 2): elif result[0] == 'append level' ***")
                list_path = result_nested['path']
                l = len(list_path)
                if l == 1:
                    tree_dict[list_path[0]].append(categoryId)
                elif l == 2:
                    tree_dict[list_path[0]][list_path[1]].append(categoryId)
                elif l == 3:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]].append(categoryId)
                elif l == 4:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]].append(categoryId)
                elif l == 5:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]][list_path[4]].append(categoryId)
                elif l == 6:
                    tree_dict[list_path[0]][list_path[1]][list_path[2]][list_path[3]][list_path[4]][list_path[5]].append(categoryId)
                # tree_dict[378][0][542h = ].append(categoryId)
            elif categoryId in tree_dict.keys():
                #print("*** 3): elif categoryId in tree_dict.keys() ***")
                tree_dict[parentId] = [tree_dict[categoryId]]
            elif parentId not in tree_dict.keys():
                #print("*** 4): elif parentId not in tree_dict.keys() ***")
                tree_dict[parentId] = [categoryId]

            elif parentId in tree_dict.keys():
                #print("*** 5): elif parentId in tree_dict.keys() ***")
                tree_dict[parentId].append(categoryId)
        i += 1
    return tree_dict, i

level_0 = 0
level_1 = 0
level_2 = 0
level_3 = 0
k = 0
list_level1 = []
dict_level1 = {}
list_level1_2 = []
dict_level1_2 = {}
## to print dictionary in a tree like structure.
def pretty(d, indent=0, parent_1=-1, parent_2=-1):
    global level_0
    global level_1
    global level_2
    global level_3
    global k
    global dict_level1
    k += 1
    for key, value in d.items():
        print('\t\t-\t' * indent + str(key))
        print("LEVEL {}".format(indent))
        if indent == 0:
            level_0 += 1
            list_level1.append(key)
            parent_1 = key
            dict_level1[parent_1] = []
        elif indent == 1:
            level_1 += 1
            if parent_1 != -1:
                dict_level1[parent_1].append(key)
        elif indent == 2:
            level_2 += 1
            if parent_1 != -1:
                dict_level1[parent_1].append(key)
        elif indent == 3:
            level_3 += 1
            if parent_1 != -1:
                dict_level1[parent_1].append(key)

        if isinstance(value, dict):
            pretty(value, indent + 1, parent_1, parent_2)
        elif isinstance(value, list):
            for i in range(0, len(value)):
                if isinstance(value[i], dict):
                    pretty(value[i], indent + 1, parent_1, parent_2)
                else:
                    print('\t\t-\t' * (indent + 1) + str(value[i]))

                    print("LEVEL {}".format(indent+1))
                    if (indent+1) == 0:
                        level_0 += 1
                        list_level1.append(value[i])
                        parent_1 = key
                        dict_level1[parent_1] = []
                    elif (indent+1) == 1:
                        level_1 += 1
                        if parent_1 != -1:
                            dict_level1[parent_1].append(value[i])
                    elif (indent+1) == 2:
                        level_2 += 1
                        if parent_1 != -1:
                            dict_level1[parent_1].append(value[i])
                    elif (indent+1) == 3:
                        level_3 += 1
                        if parent_1 != -1:
                            dict_level1[parent_1].append(value[i])

        else:
            print('\t\t-\t' * (indent+1) + str(value))
            print("LEVEL {}".format(indent+1))
            if (indent + 1) == 0:
                level_0 += 1
                parent_1 = key
                dict_level1[parent_1] = []

                list_level1.append(value)
            elif (indent + 1) == 1:
                level_1 += 1
                if parent_1 != -1:
                    dict_level1[parent_1].append(value)
            elif (indent + 1) == 2:
                level_2 += 1
                if parent_1 != -1:
                    dict_level1[parent_1].append(value)
            elif (indent + 1) == 3:
                level_3 += 1
                if parent_1 != -1:
                    dict_level1[parent_1].append(value)

pretty(category_tree()[0])
#print(dict_level1)
def items_category_level_1():
    pd.options.display.max_columns
    df_final = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
    flatten_values_l1 = []  # to make search easier
    level_1_values = list(dict_level1.values())
    level_1_keys = list(dict_level1.keys())
    for l in level_1_values:
        flatten_values_l1.extend(l)
    for index, row in df_final.iterrows():
        cat = row['value']
        #for k, v in dict_level1.items():
        if int(cat) in flatten_values_l1:
            index_value_cat = 0
            for l in level_1_values:
                if int(cat) in l:
                    substitute_cat = level_1_keys[index_value_cat]#k
                    print(substitute_cat)
                        #row['value'] = substitute_cat
                        #row['value'] = substitute_cat
                        #df_items.at[index, row['value']] = substitute_cat
                    df_final.replace([row['value']], substitute_cat, inplace=True)
                index_value_cat += 1
    print(df_final.head())
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1.csv', index=False)

#items_category_level_1()


#print(category_tree()[0])
#print(category_tree()[1])
#print("Level 1: {}".format(level_0))
#print("Level 2: {}".format(level_1))
#print("Level 3: {}".format(level_2))
#print("Level 4: {}".format(level_3))

#print(list_level1)
#print(len(list_level1))
#print(category_tree())
#print(dict_level1)
#list_keys_sorted = list(dict_level1.keys())
#print(sorted(list_keys_sorted))



## to replace categoryid values in dataset by their parent category from level 2 or 1.
def items_category_level_1_2():
    new_dict = category_tree()[0]
    values_level_1 = list(new_dict.values())
    i = 0
    values_level_1_2 = []
    dict_level2 = {}
    flatten_values = []  # to make search easier
    x = 0
    for l in values_level_1:
        for val in l:
            if isinstance(val, dict):
                level_2_parent = list(val.keys())[0]
                temp_values_level_2 = list(val.values())
                dict_level2[level_2_parent] = temp_values_level_2[0]
                j = 0
                for val_level_2 in temp_values_level_2[0]:
                    if isinstance(val_level_2, dict):
                        del dict_level2[level_2_parent][j]
                        to_extend_keys_level3 = list(val_level_2.keys())
                        dict_level2[level_2_parent].extend(to_extend_keys_level3)
                        temp_values_level_3 = list(val_level_2.values())  # to extend, values from level 3
                        dict_level2[level_2_parent].extend(temp_values_level_3[0])
                        flatten_values.extend(to_extend_keys_level3)
                        for val_level_3 in temp_values_level_3[0]:
                            if isinstance(val_level_3, dict):
                                to_extend_keys_level4 = list(val_level_3.keys())
                                dict_level2[level_2_parent].extend(to_extend_keys_level4)
                                temp_values_level_4 = list(val_level_3.values())
                                dict_level2[level_2_parent].extend(temp_values_level_4)
                                flatten_values.extend(temp_values_level_4[0])
                                flatten_values.extend(to_extend_keys_level4)
                            else:
                                flatten_values.append(val_level_3)
                    else:
                        flatten_values.append(val_level_2)
                    j += 1
            else:
                values_level_1_2.append(val)

        i += 1

    print("keys level 1)......")
    keys_level_1 = list(new_dict.keys())
    # print(keys_level_1)
    # print("values level 1_2).......")
    # print(values_level_1_2)
    # print("dict level 2...")
    # print(dict_level2)
    keys_level2 = list(dict_level2.keys())
    values_level2 = list(dict_level2.values())
    # print(keys_level2)
    # print(values_level2)
    # print("flatten...")
    # print(flatten_values)
    pd.options.display.max_columns
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/bla/item_properties.csv')
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/bla/events.csv')
    df_items.rename(columns={'timestamp': 'timestamp_2'}, inplace=True)
    df_items = df_items[df_items['property'] == 'categoryid']
    print(df_items.head(10))
    for index, row in df_items.iterrows():
        cat = row['value']
        #if int(cat) in keys_level_1 -> all right. already level 1.
        #if int(cat) in values_level_1_2 -> all right. already level 1 or 2.
        #if int(cat) in keys_level2 -> all right. already level 2.
        if int(cat) in flatten_values:
            index_value_cat = 0
            for li in values_level2:
                if int(cat) in li:
                    substitute_cat = keys_level2[index_value_cat]
                    df_items.replace([row['value']], substitute_cat, inplace=True)
                index_value_cat += 1

    print(df_items.head())
    df_items.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level2/joined_events_items_level2.csv',
                    index=False)

#items_category_level_1_2()


# group sessions with level1 categories
def group_sessions_cats_level1():
    df_final = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1.csv')
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
    df_final = df_final[['visitorid', 'timestamp', 'event', 'itemid', 'transactionid', 'property', 'value']]
    df_final.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv', index=False)

group_sessions_cats_level1()

## simple statistical analysis of the newly created dataset 'joined_events_items_longer_seqs'
def experiments_groupings_cats_level1():
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
def dataset_split_users_cats_level1():
    df_visitors = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_events_items_level1_longer_seqs.csv')
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

    ## guardar os diferentes datasets!
    df_train.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_train.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_train_level1.csv',
                    index=False)
    df_test.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_test.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_level1/joined_test_level1.csv',
                   index=False)

dataset_split_users_cats_level1()

## function to shift the dataset in order to be in the form: per row, X sequence of categories (taken from each event) and y next real category
## -> result: csv files 'shifted_train' and 'shifted_test'
## includes, a minmaxscaling of the categories as well, and this experiment is saved in csv files 'shifted_train_cat_scaling' and 'shifted_test_cat_scaling'
def prepare_dataset_seqs_target_cats_level1():
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