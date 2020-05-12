import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
from pprint import pprint

pd.options.display.max_columns

df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items_new.csv')

highest_category = 1696
range_categories = 290

# print(dict_categories)

def get_transition_matrix_re_use(sequence): # sequence of items' categories in history
    dict_categories = dict.fromkeys(range(highest_category + 1), 0)

    n = 1 + highest_category # max(sequence)  # number of states

    M_counts = np.zeros([n, n], dtype=float)  # creating initial matrix of all-zeros

    for (i, j) in zip(sequence, sequence[1:]):  # to connect "sequences" of 2 events, followed by each other -> matriz de contagens
        M_counts[i][j] += 1
        #print("M[{{0:.5f}}][{{0:.5f}}] = {}".format(i, j, M[i][j]))
        dict_categories[i] += 1 #count number of times it starts from i
    print("in M_counts[859][0] = {}".format(M_counts[859][0]))
    print("in M_counts[859][859] = {}".format(M_counts[859][859]))

    #for r in M_counts: print(' '.join('{0:.9f}'.format(x) for x in r))
    print("in dict_categories[955] = {}".format(dict_categories[859]))
    print("in dict_categories[0] = {}".format(dict_categories[0]))

    # go trough matrix, to convert to probabilities -> using Laplace smoothing:
    M_probs = M_counts.copy()
    for i in range(len(M_probs)):
        for j in range(len(M_probs[i])):
            # print(dict_categories[i])
            M_probs[i][j] = (M_probs[i][j] + 1) / (dict_categories[i] + range_categories)

    #for r in M_probs: print(' '.join('{0:.5f}'.format(x) for x in r))
    print("in M_probs[859][0] = {}".format(M_probs[859][0]))
    print("in M_probs[859][859] = {}".format(M_probs[859][859]))
    print("max...{} ".format((M_probs[859]).argmax()))
    return M_probs

def get_transition_matrix(sequence, list_indexes): # sequence of items' categories in history
    distinct_categories_ = len(list_indexes)
    n = distinct_categories_ # max(sequence)  # number of states PENSAR!!!!!!

    #dict_categories = dict.fromkeys(range(highest_category + 1), 0)
    dict_categories = dict.fromkeys(list_indexes, 0)
    M_counts = np.zeros([n, n], dtype=float)  # creating initial matrix of all-zeros

    for (i, j) in zip(sequence, sequence[1:]):  # to connect "sequences" of 2 events, followed by each other -> matriz de contagens
        index_i = list_indexes.index(i)
        #print("----------")
        #print(i)
        #print(index_i)
        index_j = list_indexes.index(j)
        #print(j)
        #print(index_j)
        M_counts[index_i][index_j] += 1
        #print("M[{{0:.5f}}][{{0:.5f}}] = {}".format(i, j, M[i][j]))
        dict_categories[i] += 1 #count number of times it starts from i
    for r in M_counts: print(' '.join('{}'.format(x) for x in r))
    print(dict_categories)

    M_probs = M_counts.copy()
    # go trough matrix, to convert to probabilities -> using Laplace smoothing:
    for i in range(len(M_probs)):
        for j in range(len(M_probs[i])):
            #print(list_indexes[i])
            M_probs[i][j] = (M_probs[i][j] + 1) / (dict_categories[list_indexes[i]] + distinct_categories_)

    for r in M_probs: print(' '.join('{0:.5f}'.format(x) for x in r))

    return M_probs

def train_model():
    df_train = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_train.csv')

    unique_visitors = df_train['visitorid'].unique()
    right = 0
    nr_cases = 0
    for current_visitor in unique_visitors:
        y = []
        for index, row in df_train.iterrows():
            #print("______" + str(i))
            categoryId = int(row['value'])
            visitorId = int(row['visitorid'])
            if visitorId == current_visitor:
                y.append(categoryId)
        print("------------------------------------")
        print("visitor: {}".format(current_visitor))
        print(y)
        compare_cat = y.pop()
        last_category = y[-1]
        print(last_category)
        #line_transition_from_last_cat = m[last_category]
        print("- def get_transition_matrix ")
        list_indexes = list(set(y))
        list_indexes.sort() #necessário mm? **
        #print(list_indexes)
        m = get_transition_matrix(y, list_indexes)
        going_state_index = np.argmax(np.amax(m[list_indexes.index(last_category)], axis=0))
        print("predicted next category: {}; real next category: {}".format(list_indexes[going_state_index], compare_cat))
        if list_indexes[going_state_index] == compare_cat:
            right += 1
        nr_cases += 1
    accuracy = right/nr_cases
    print("accuracy = {}; nr cases gotten right = {}; total nr of cases = {}".format(accuracy, right, nr_cases))

    '''print("----- def get_transition_matrix -----")
    m_2 = get_transition_matrix(y)
    going_state_2 = (m_2[last_category]).argmax()
    print("predicted next category: {}; real next category: {}".format(going_state_2, compare_cat))'''

train_model()


def train_get_result():
    df.sort_values(by=['timestamp'], inplace=True)
    y = df['value']
    X = df.drop(['value'], axis=1)
    train_pct_index = int(0.1 * len(y))
    #X_train, X_test = X[:train_pct_index], X[train_pct_index:]
    #y_train, y_test = y[:train_pct_index], y[train_pct_index:]
    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
    #print(y_train.head(30))

    y_train = y
    m = get_transition_matrix_re_use(y_train)
    #for r in m: print(' '.join('{0:.9f}'.format(x) for x in r))
    last_category = y_train.iloc[[-1]]
    last_category = int(last_category.to_string(index=False))
    print("last category: %s" % last_category)
    line_transition_from_last_cat = m[last_category]
    #print(line_transition_from_last_cat)
    k = 0
    for i in line_transition_from_last_cat:
        print(k)
        print(i)
        k+=1
    print(line_transition_from_last_cat[1095])
    print(np.argmax(line_transition_from_last_cat, axis=0))
    #print(line_transition_from_last_cat.index(max(line_transition_from_last_cat)))
    going_state = np.argmax(np.amax(m[last_category], axis=0))       # np.amax(m[last_category])# np.argmax(np.max(m[last_category], axis=0))
    #print("prediction ... it is more probable a transition FROM the last state {} TO {}".format(last_category, going_state))











################### TRANSITIONS GRAPH - EXPERIMENTS

# create a function that maps transition probability dataframe
# to markov edges and weights

def get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


def get_transitions_diagram(m):
    edges_wts = get_markov_edges(pd.DataFrame(m))
    pprint(edges_wts)

    # create graph object
    G = nx.MultiDiGraph()
    states = [0, 1, 2]
    # nodes correspond to states
    G.add_nodes_from(states)
    # print(f'Nodes:\n{G.nodes()}\n')

    # edges represent transition probabilities
    for k, v in edges_wts.items():
        tmp_origin, tmp_destination = k[0], k[1]
        G.add_edge(tmp_origin, tmp_destination, weight=v, label=v)
    # print(f'Edges:')
    pprint(G.edges(data=True))

    pos = nx.drawing.nx_pydot.graphviz_layout(G, prog='dot')
    nx.draw_networkx(G, pos)
    '''
    # create edge labels for jupyter plot but is not necessary
    edge_labels = {(n1,n2):d['label'] for n1,n2,d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G , pos, edge_labels=edge_labels)
    nx.drawing.nx_pydot.write_dot(G, '/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/retail_rocket_markov_chain.dot')
    nx.draw(G)
    plt.show()'''
