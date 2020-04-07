import pandas as pd
from collections import defaultdict
import numpy as np
from anytree.exporter import DotExporter

df_cat = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/category_tree.csv')

df_cat = df_cat.fillna(-1)
df_cat['parentid'] = df_cat['parentid'].astype(int)

# print(df_cat.head(25))


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

            #print("AQUELA KEY: {}".format(result['key']))
            #print("CURRENT KEY: {}".format(key))
            #result['path'] = []

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
def pretty(d, indent=0, p=-1):
    global level_0
    global level_1
    global level_2
    global level_3
    global k
    global dict_level1
    k += 1
    for key, value in d.items():
        print('\t\t-\t' * indent + str(key))
        #print("LEVEL {}".format(indent))
        if indent == 0:
            level_0 += 1
            list_level1.append(key)
            #if p == -1:
            p = key
            dict_level1[p] = []

        elif indent == 1:
            level_1 += 1
            if p != -1:
                dict_level1[p].append(key)

        elif indent == 2:
            level_2 += 1
            if p != -1:
                dict_level1[p].append(key)
        elif indent == 3:
            level_3 += 1
            if p != -1:
                dict_level1[p].append(key)

        if isinstance(value, dict):
            pretty(value, indent+1, p)
        elif isinstance(value, list):
            for i in range(0, len(value)):
                if isinstance(value[i], dict):
                    pretty(value[i], indent + 1, p)
                else:
                    print('\t\t-\t' * (indent + 1) + str(value[i]))

                    #print("LEVEL {}".format(indent+1))
                    if (indent+1) == 0:
                        level_0 += 1
                        list_level1.append(value[i])
                        #if p == -1:
                        p = key
                        dict_level1[p] = []
                    elif (indent+1) == 1:
                        level_1 += 1
                        if p != -1:
                            dict_level1[p].append(value[i])

                    elif (indent+1) == 2:
                        level_2 += 1
                        if p != -1:
                            dict_level1[p].append(value[i])

                    elif (indent+1) == 3:
                        level_3 += 1
                        if p != -1:
                            dict_level1[p].append(value[i])

        else:
            print('\t\t-\t' * (indent+1) + str(value))
            #print("LEVEL {}".format(indent))

            if (indent + 1) == 0:
                level_0 += 1
                #if p == -1:
                p = key
                dict_level1[p] = []

                list_level1.append(value)
            elif (indent + 1) == 1:
                level_1 += 1
                if p != -1:
                    dict_level1[p].append(value)
            elif (indent + 1) == 2:
                level_2 += 1
                if p != -1:
                    dict_level1[p].append(value)
            elif (indent + 1) == 3:
                level_3 += 1
                if p != -1:
                    dict_level1[p].append(value)
            #print("LEVEL {}".format(indent+1))

pretty(category_tree()[0])
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

def items_category_level_1():
    pd.options.display.max_columns
    df_items = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/bla/item_properties.csv')
    df_events = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/bla/events.csv')
    df_items.rename(columns={'timestamp': 'timestamp_2'}, inplace=True)
    df_items = df_items[df_items['property'] == 'categoryid']
    print(df_items.head(10))
    i = 0
    for index, row in df_items.iterrows():
        cat = row['value']
        for k, v in dict_level1.items():
            if int(cat) in v:
                substitute_cat = k
                print(substitute_cat)
                #row['value'] = substitute_cat
                #row['value'] = substitute_cat
                #df_items.at[index, row['value']] = substitute_cat
                df_items.replace([row['value']], substitute_cat, inplace=True)
        i += 1
    print(df_items.head())
    df_items.to_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/item_properties_modified.csv', index=False)

#items_category_level_1()