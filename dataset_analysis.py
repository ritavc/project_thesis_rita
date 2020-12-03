import pandas as pd
import datetime
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

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

    # print(df_final.shape)
    unique_visitors = df_final['visitorid'].unique()
    # print(unique_visitors)

    # print(df_final.head())
    # print("maximum cat: %s " % np.amax(df_final.value.astype(int)))
    # df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv', index=False)
    return unique_visitors


df_final_creation()


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

def statistics():
    df = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv')
    df = df[(df.level2 != int(-1))]
    users = df['visitorid'].unique()
    print("Distinct nr of users: %s" % (users.shape,))
    items = df['itemid'].unique()
    print("Distinct nr of items: %s" % (items.shape,))
    cats_items_dict_2 = {}
    cats_items_dict_1 = {}

    for index, row in df.iterrows():
        if int(row['level2']) in cats_items_dict_2:
            if int(row['itemid']) not in cats_items_dict_2.get(int(row['level2'])):
                cats_items_dict_2[int(row['level2'])].append(int(row['itemid']))
        else:
            cats_items_dict_2[int(row['level2'])] = [int(row['itemid'])]
        if int(row['level1']) in cats_items_dict_1:
            if int(row['itemid']) not in cats_items_dict_1.get(int(row['level1'])):
                cats_items_dict_1[int(row['level1'])].append(int(row['itemid']))
        else:
            cats_items_dict_1[int(row['level1'])] = [int(row['itemid'])]
    # new_dict_2 = {k: v for k, v in sorted(cats_items_dict_2.items(), key=lambda item: item[1])}
    # new_dict_1 = {k: v for k, v in sorted(items_cats_dict_1.items(), key=lambda item: item[1])}
    # print(new_dict_2)
    # print("****")
    # print(new_dict_1)

    for key_2, value_2 in cats_items_dict_2.items():
        cats_items_dict_2[key_2] = len(value_2)
    new_dict_2 = {k: v for k, v in sorted(cats_items_dict_2.items(), key=lambda item: item[1])}
    print(new_dict_2)
    print(len(new_dict_2))
    plt.barh(range(len(new_dict_2)), new_dict_2.values(), align='center')
    plt.xlabel('amount of corresponding items', fontsize=10)
    plt.ylabel('categories level-2', fontsize=10)
    ax = plt.gca()
    ax.axes.yaxis.set_ticks([])
    plt.show()
    items_cats_dict_2 = {}
    for index, row in df.iterrows():
        if int(row['itemid']) not in cats_items_dict_2:
            cats_items_dict_2[int(row['itemid'])] = [int(row['level2'])]
    # print(cats_items_dict_2)
    print(df["event"].value_counts())
    print(df.shape)


# statistics()

def plot_lengths():
    axis_x_window_length = [2, 3, 4, 5]
    axis_y_nr_sessions = [26141, 7362, 2596, 1212]
    axis_y_results = [0.464, 0.280, 0.113, 0.020]


    data_plot = pd.DataFrame({"sessions length": axis_x_window_length, "number of sessions": axis_y_nr_sessions})
    data_plot_2 = pd.DataFrame({"sessions length": axis_x_window_length, "recall@10": axis_y_results})

    '''plt.bar(axis_x_window_length, axis_y_nr_sessions, align='center')
    plt.xlabel('sessions length', fontsize=10)
    plt.ylabel('number of sessions', fontsize=10)
    plt.scatter(axis_x_window_length, axis_y_results)
    plt.plot(axis_x_window_length, axis_y_results, linestyle='dashed')
    plt.show()'''
    # Create combo chart
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # bar plot creation
    ax1.set_xlabel('sessions length', fontsize=10)
    ax1.set_ylabel('number of sessions', fontsize=10)
    ax1 = sns.barplot(x='sessions length', y='number of sessions', data=data_plot, color='lightblue')

    #ax1.tick_params(axis='y')

    # specify we want to share the same x-axis
    ax2 = ax1.twinx()
    # line plot creation
    ax2.set_ylabel('recall@10', fontsize=10)
    ax2 = sns.pointplot(x='sessions length', y='recall@10', data=data_plot_2, linestyles="--", color='grey')
    #ax2.tick_params(axis='y', color=color)
    # show plot
    #plt.show()
plot_lengths()
def to_delete():
    df_original = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_cat_levels.csv')
    print("before any duplicates removal: " + str(df_original.shape))
    df_no_duplicates = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_no_duplicates.csv')
    print("after any duplicates removal: " + str(df_no_duplicates.shape))
    print(df_no_duplicates['level2'].value_counts())
    print(df_no_duplicates['level3'].value_counts())
    df_after_division_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/train_set.csv')
    df_after_division_val = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/validation_set.csv')
    df_after_division_test = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/test_set.csv')
    print("train set after division : " + str(df_after_division_train.shape))
    print("validation set after division : " + str(df_after_division_val.shape))
    print("test set after division : " + str(df_after_division_test.shape))
    print("total after division: " + str(
        df_after_division_train.shape[0] + df_after_division_val.shape[0] + df_after_division_test.shape[0]))
    df_sessions_train = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_train_set.csv')
    df_sessions_val = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_val_set.csv')
    df_sessions_test = pd.read_csv(
        '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/fixed_test_set.csv')
    print("train set after sessions creation (>3min inactivity): " + str(df_sessions_train.shape))
    print("validation set after sessions creation (>3min inactivity): " + str(df_sessions_val.shape))
    print("test set after sessions creation (>3min inactivity): " + str(df_sessions_test.shape))
    print("total after sessions creation: " + str(
        df_sessions_train.shape[0] + df_sessions_val.shape[0] + df_sessions_test.shape[0]))

# to_delete()

# timelag_sessions_division()
# prepare_dataset_seqs_target()
# dataset_split_users()
# dataset_split_time()
# experiments_groupings()
# statistical_analysis_events()
# statistical_analysis_items()
