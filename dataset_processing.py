import pandas as pd
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

'''
df_items.rename(columns={'timestamp': 'timestamp_2'}, inplace=True)
df_items = df_items[(df_items['property'] == 'categoryid')]
df_final = df_events.merge(df_items, on=['itemid'])
df_final.drop(['timestamp_2', 'id'], axis=1, inplace=True)
df_final.sort_values(by=['timestamp'], inplace=True)
# print(df_final.head())
# print("maximum cat: %s " % np.amax(df_final.value))
# df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')'''
pd.options.display.max_columns
pd.set_option('display.max_columns', None)


def item_properties_concat():
    df1 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part1.csv")
    df2 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part2.csv")

    df = pd.concat([df1, df2], names=['timestamp', 'itemid', 'property', 'value'])
    df = df[['timestamp', 'itemid', 'property', 'value']]
    #df.sort_values(by=['itemid'], inplace=True)
    #print(df.head(50))
    #df.drop(df.columns[[0]], axis=1, inplace=True)
    df.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')

# item_properties_concat()
def statistical_analysis_events():
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')

    # print(df_events['event'].value_counts())
    print("Nr of instances in events dataset: %s "%(df_events.shape,))
    df_events.sort_values(by=['timestamp'], inplace=True)

    visitors = df_events["visitorid"].unique()
    print("Unique sessions/visitors: %s"%(visitors.shape,))
    print('Total sessions/visitors: ', df_events['visitorid'].size)
    print(df_events['event'].value_counts())
    print(df_events.loc[(df_events.itemid == 218)].sort_values('timestamp').head(30))


def statistical_analysis_items():
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')

    df_items.drop(['Unnamed: 0'], axis=1, inplace=True)
    print(df_events['event'].value_counts())
    print("Nr of instances in items dataset: %s "%(df_items.shape,))
    df_items.sort_values(by=['itemid', 'timestamp'], ascending=[True, True], inplace=True)

    items = df_events["itemid"].unique()
    print("Unique items: %s"%(items.shape,))

    times = []
    for i in df_items['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    #df_items['timestamp'] = times
    # print(df_items.head())
    # print(df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head(10))
    # print(df_items['itemid'].head())
    df_items_ = df_items[df_items['property'] == 'categoryid']
    #df_items_.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/items_properties_categories.csv')

    print(df_items_.head(50))
    '''print(df_items.loc[(df_items.itemid == 206783) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 264319) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 178601) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 346165) & (df_items.property == 'categoryid')])'''

def edit_items_dataset():
    df_items_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_modified.csv')
    groups_items = df_items_cat.groupby('itemid')
    #for key, item in groups_items:
     #   print(item)
    df_items_cat.drop_duplicates(['itemid'], inplace=True)
    df_items_cat.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_modified.csv', index=False)

#edit_items_dataset()

def df_final_creation():
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/bla/events.csv')
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_modified.csv')

    df_final = df_events.merge(df_items, on=['itemid'])
    df_final.drop(['timestamp_2', 'id'], axis=1, inplace=True)
    df_final.sort_values(by=['timestamp'], inplace=True)

    print(df_final.head())
    #print("maximum cat: %s " % np.amax(df_final.value.astype(int)))
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv', index=False)

#df_final_creation()



def statistical_analysis_category_tree():
    df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')

    # print(df_events['event'].value_counts())
    print("Nr of instances in category tree dataset: %s" % (df_cat.shape, ))
    categories = df_cat['categoryid'].unique()
    print("Distinct nr of categories: %s"%(categories.shape,))
    print("number of items under category id 1016: ")
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')

    print(df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head())

    print("maximum and minimum values of categories: ")
    print("maximum: %s "%np.amax(df_cat.categoryid))
    print("minimum: %s " % np.amin(df_cat.categoryid))


def group_sessions():
    df_final = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')
    print("Nr of instances in joined dataset: %s "%(df_final.shape,))

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
    df_final.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_new.csv', index=False)

group_sessions()

#statistical_analysis_events()
#statistical_analysis_items()
