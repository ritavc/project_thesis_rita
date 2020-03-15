import pandas as pd
import datetime
import numpy as np

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


df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')
df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')
df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/events.csv')


def statistical_analysis_events():
    # print(df_events['event'].value_counts())
    print("Nr of instances in events dataset: %s "%(df_events.shape,))
    df_events.sort_values(by=['timestamp'], inplace=True)

    visitors = df_events["visitorid"].unique()
    print("Unique sessions/visitors: %s"%(visitors.shape,))
    print('Total sessions/visitors: ', df_events['visitorid'].size)
    print(df_events['event'].value_counts())


def statistical_analysis_items():
    # print(df_events['event'].value_counts())
    print("Nr of instances in items dataset: %s "%(df_items.shape,))
    df_items.sort_values(by=['itemid'], inplace=True)

    # items = df_events["itemid"].unique()
    # print("Unique items: %s"%(items.shape,))

    times = []
    for i in df_items['timestamp']:
        times.append(datetime.datetime.fromtimestamp(i // 1000.0))
    df_items['timestamp'] = times
    # print(df_items.head())
    # print(df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head(10))
    # print(df_items['itemid'].head())

    print(df_items.loc[(df_items.itemid == 206783) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 264319) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 178601) & (df_items.property == 'categoryid')])
    print(df_items.loc[(df_items.itemid == 346165) & (df_items.property == 'categoryid')])



def statistical_analysis_category_tree():
    # print(df_events['event'].value_counts())
    print("Nr of instances in category tree dataset: %s" % (df_cat.shape, ))
    categories = df_cat['categoryid'].unique()
    print("Distinct nr of categories: %s"%(categories.shape,))
    print("number of items under category id 1016: ")
    print(df_items.loc[(df_items.property == 'categoryid') & (df_items.value == '1016')].sort_values('timestamp').head())


# statistical_analysis_category_tree()
# statistical_analysis_events()
statistical_analysis_items()