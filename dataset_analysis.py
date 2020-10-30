import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

pd.options.display.max_columns
pd.set_option('display.max_columns', None)


## function that concatenates item_properties part 1 and 2 data-sets provided in the Kaggle repository files.
# -> result: csv file 'item_properties'
def item_properties_concat():
    df1 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part1.csv")
    df2 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part2.csv")

    df = pd.concat([df1, df2], names=['timestamp', 'itemid', 'property', 'value'])
    df = df[['timestamp', 'itemid', 'property', 'value']]
    # df.sort_values(by=['itemid'], inplace=True)
    # print(df.head(50))
    # df.drop(df.columns[[0]], axis=1, inplace=True)
    df.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv', index=False)


## simple statistical analysis of the events dataset
def statistical_analysis_events():
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')
    times = []
    for i in df_events['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    df_events['timestamp'] = times
    # print(df_events['event'].value_counts())
    print("Nr of instances in events dataset: %s " % (df_events.shape,))
    df_events.sort_values(by=['timestamp'], inplace=True)

    visitors = df_events["visitorid"].unique()
    print("Unique sessions/visitors: %s" % (visitors.shape,))
    print('Total sessions/visitors: ', df_events['visitorid'].size)
    print(df_events['event'].value_counts())
    print(df_events.loc[(df_events.itemid == 218)].sort_values('timestamp').head(30))
    print("first element time:")
    print(df_events['timestamp'].iloc[0])  # first element
    print("last element time:")
    print(df_events['timestamp'].iloc[-1])  # last element


## simple statistical analysis of the items dataset AND filtering of the items that have as 'property' == 'categoryid' (the value to be predicted in this problem.)
# -> result: csv file 'items_properties_categories'
def statistical_analysis_items():
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')

    df_items.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(df_events['event'].value_counts())
    print("Nr of instances in items dataset: %s " % (df_items.shape,))
    df_items.sort_values(by=['itemid', 'timestamp'], ascending=[True, True], inplace=True)

    items = df_events["itemid"].unique()
    print("Unique items: %s" % (items.shape,))

    times = []
    for i in df_items['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    # df_items['timestamp'] = times
    # print(df_items.head())
    # print(df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head(10))
    # print(df_items['itemid'].head())
    df_items_ = df_items[df_items['property'] == 'categoryid']
    df_items_.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/items_properties_categories.csv',
                     index=False)

    print(df_items_.head(50))
    '''print(df_items.loc[(df_items.itemid == 206783) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 264319) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 178601) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 346165) & (df_items.property == 'categoryid')])'''


## simple statistical analysis of the category_tree dataset
def statistical_analysis_category_tree():
    df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')

    # print(df_events['event'].value_counts())
    print("Nr of instances in category tree dataset: %s" % (df_cat.shape,))
    categories = df_cat['categoryid'].unique()
    print("Distinct nr of categories: %s" % (categories.shape,))
    print("number of items under category id 1016: ")
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')

    print(
        df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head())

    print("maximum and minimum values of categories: ")
    print("maximum: %s " % np.amax(df_cat.categoryid))
    print("minimum: %s " % np.amin(df_cat.categoryid))


## to solve the issue of having multiple items with varying details over time, only the first item's details registered will be considered.
# -> result: csv file 'item_properties_original'
def edit_items_dataset_original_details():
    df_items_cat = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/items_properties_categories.csv')
    groups_items = df_items_cat.groupby('itemid')
    # for key, item in groups_items:
    #   print(item)
    df_items_cat.drop_duplicates(['itemid'], inplace=True)
    df_items_cat.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_original.csv',
                        index=False)


## join events dataset with the items (with categoryid filtered) dataset.
# -> result: csv file 'joined_events_items'
def df_final_creation():
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_original.csv')
    df_items.columns = ['timestamp_2', 'itemid', 'property', 'value']
    df_final = df_events.merge(df_items, on=['itemid'])
    df_final.drop(['timestamp_2'], axis=1, inplace=True)
    df_final.sort_values(by=['timestamp'], inplace=True)

    print(df_final.head())
    # print("maximum cat: %s " % np.amax(df_final.value.astype(int)))
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv', index=False)


## grouping sessions by visitors, then check the length of events' sequences in time performed by each user.
# drop user's sequences of events shorter than 3 timesteps.
## -> result: csv file 'joined_events_items_longer_seqs'
def group_sessions():
    df_final = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
    print("Nr of instances in joined dataset: %s " % (df_final.shape,))

    # print("maximum cat: %s " % np.amax(df_final.value.astype(int)))
    # print((df_final['value'].unique()).shape)

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
        # print(i, row)
    print("bigger: " + str(bigger))
    print("smaller: " + str(smaller))
    axs_x = ['shorter sequences w/ less or equal to 3 events', 'seq w/ more than 3 events']
    axs_y = [smaller, bigger]
    sns.barplot(axs_x, axs_y)  # , ax=axs[0])
    # plt.show()
    print(list_visitorId_consider)
    df_final = df_final[df_final['visitorid'].isin(list_visitorId_consider)]
    # print(df_final.head(30))
    df_final = df_final[['visitorid', 'timestamp', 'event', 'itemid', 'transactionid', 'property', 'value']]
    df_final.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_longer_seqs.csv',
                    index=False)


# group_sessions()

## simple statistical analysis of the newly created dataset 'joined_events_items_longer_seqs'
def experiments_groupings():
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_longer_seqs.csv')
    groups = df_visitors.groupby('visitorid')
    print("Nr of instances in dataset: %s" % (df_visitors.shape,))
    # print(groups.size().value_counts().head(60))
    print("maximum: %s " % np.amax(df_visitors.value))
    print("minimum: %s " % np.amin(df_visitors.value))
    categories = df_visitors['value'].unique()
    print("Distinct nr of categories: %s" % (categories.shape,))
    users = df_visitors['visitorid'].unique()
    print("Distinct nr of users: %s" % (users.shape,))


## experiments to split the dataset into train and test sets. division done in time (choosing a point in time to divide)
# includes different division points.
# includes simple statistical analysis to check how many short/long sequences result in each division performed.
# the goal here was to divide the users' sequences in time -> to train on the first part of the sequence and test on the other part on the same user.
# however, the sequences turned out to be broken in too small sequences. proved out to be not a good idea then.
## -> result: csv files 'joined_train' and 'joined_test'
def dataset_split_time():
    df_visitors = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_longer_seqs.csv')
    df_visitors.sort_values(by=['timestamp'], inplace=True)
    train_pct_index = int(0.55 * len(df_visitors))
    # X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    # y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    df_train, df_test = df_visitors[:train_pct_index], df_visitors[train_pct_index:]
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

    unique_visitors_train = df_train['visitorid'].unique()
    unique_visitors_test = df_test['visitorid'].unique()
    common_visitors = list(set(unique_visitors_train).intersection(list(unique_visitors_test)))
    print(len(common_visitors))

    new_train = df_train[df_train['visitorid'].isin(common_visitors)]
    new_test = df_test[df_test['visitorid'].isin(common_visitors)]

    groups_train_unique = new_train.sort_values(['visitorid'], ascending=True).groupby('visitorid')
    groups_size_train_unique = groups_train_unique.size()
    print("groups train: ")
    print(groups_size_train_unique)
    bigger_train_unique = 0
    smaller_train_unique = 0
    list_long_train_unique = []
    for i, row in groups_size_train_unique.iteritems():
        if int(row) > 3:
            bigger_train_unique += 1
            list_long_train_unique.append(i)
        else:
            smaller_train_unique += 1
        # print(i, row)

    print("number of smaller seqs in train unique : {}".format(smaller_train_unique))
    print("number of bigger seqs in train unique : {}".format(bigger_train_unique))

    groups_test_unique = new_test.sort_values(['visitorid'], ascending=True).groupby('visitorid')
    groups_size_test_unique = groups_test_unique.size()
    print("groups test: ")
    print(groups_size_test_unique)
    bigger_test_unique = 0
    smaller_test_unique = 0
    list_long_test_unique = []
    list_small_test_unique = []
    for i, row in groups_size_test_unique.iteritems():
        if int(row) > 3:
            bigger_test_unique += 1
            list_long_test_unique.append(i)
        else:
            smaller_test_unique += 1
            list_small_test_unique.append(i)

        # print(i, row)
    print("number of smaller seqs in test : {}".format(smaller_test_unique))
    print("number of bigger seqs in test : {}".format(bigger_test_unique))

    common_visitors_big = list(set(list_long_train_unique).intersection(list(list_long_test_unique)))
    print("number of visitors with big seqs in both sides: ")
    print(len(common_visitors_big))

    common_visitors_big_small = list(set(list_long_train_unique).intersection(list(list_small_test_unique)))
    print("number of visitors with big seqs in train and small in test: ")
    print(len(common_visitors_big_small))

    '''df_train.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_train.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_train.csv',
                    index=False)
    df_test.sort_values(['visitorid', 'timestamp'], ascending=[True, True], inplace=True)
    df_test.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_test.csv',
                   index=False)'''


# timelag_sessions_division()
# prepare_dataset_seqs_target()
# dataset_split_users()
# dataset_split_time()
# experiments_groupings()
statistical_analysis_events()
# statistical_analysis_items()
