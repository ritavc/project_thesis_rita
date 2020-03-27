import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
from pprint import pprint

df = pd.read_csv('/Users/ritavconde/Documents/MEIC-A/Tese/ecommerce-dataset/joined_events_items.csv')

maximum_categories = 1698

dict_categories = dict.fromkeys(range(maximum_categories+1), 0)
# print(dict_categories)

def get_transition_matrix(sequence): # sequence of items' categories in history
    n = 1 + maximum_categories # max(sequence)  # number of states

    M = np.zeros([n, n], dtype=float)  # creating initial matrix of all-zeros

    for (i, j) in zip(sequence, sequence[1:]):  # to connect "sequences" of 2 events, followed by each other
        M[i][j] += 1
        dict_categories[i] += 1
    # print(dict_categories)

    # go trough matrix, to convert to probabilities -> using Laplace smoothing:
    for i in range(len(M)):
        for j in range(len(M[i])):
            # print(dict_categories[i])
            M[i][j] = (M[i][j] + 1) / (dict_categories[i] + maximum_categories)

    return M
df.sort_values(by=['timestamp'], inplace=True)
y = df['value']
X = df.drop(['value'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.7)
'''
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

'''
m = get_transition_matrix(y_train)
# for r in m: print(' '.join('{0:.5f}'.format(x) for x in r))
last_category = y_train.iloc[[-1]]
print("last category: %s" % last_category)
going_state = np.argmax(np.amax(m[last_category], axis=0))       # np.amax(m[last_category])# np.argmax(np.max(m[last_category], axis=0))
print("prediction ... it is more probable a transition FROM the last state {} TO {}".format(last_category.to_string(index=False), going_state))

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
