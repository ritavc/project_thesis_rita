import pandas as pd

df1 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part1.csv")
df2 = pd.read_csv("/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_part2.csv")

df = pd.concat([df1, df2], names=['timestamp', 'itemid', 'property', 'value'])
df = df[['itemid', 'timestamp', 'property', 'value']]
df.sort_values(by=['itemid'], inplace=True)
#print(df.head(50))
df.drop(df.columns[[0]], axis=1, inplace=True)
df.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties.csv')