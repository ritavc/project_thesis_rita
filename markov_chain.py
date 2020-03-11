import matplotlib
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from pprint import pprint

df_events = pd.read_csv('ecommerce-dataset/events.csv')
# print(df_events['event'].value_counts())

df_events.sort_values(by=['timestamp'], inplace=True)
# encoding categorical to numerical. states: view, transaction, addtocart

df_events['event'] = df_events['event'].astype('category')
df_events['event_cat'] = df_events['event'].cat.codes

y = df_events['event_cat']
X = df_events.drop(['event_cat', 'event'], axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# print(df_events.head(20))
# print(X_train.head())
# print(y_train.head())


def get_transition_matrix(sequence):
    n = 1 + max(sequence)  # number of states

    total_trans_from_0 = 0
    total_trans_from_1 = 0
    total_trans_from_2 = 0

    M = np.zeros([n, n], dtype=float)  # creating initial matrix of all-zeros

    for (i, j) in zip(sequence, sequence[1:]):  # to connect "sequences" of 2 events, followed by each other
        M[i][j] += 1
        if i == 0:
            total_trans_from_0 += 1
        elif i == 1:
            total_trans_from_1 += 1
        elif i == 2:
            total_trans_from_2 += 1

    # go trough matrix, to convert to probabilities -> using Laplace smoothing:
    for i in range(len(M)):
        for j in range(len(M[i])):
            if i == 0:
                M[i][j] = (M[i][j]+1)/(total_trans_from_0+3)
            elif i == 1:
                M[i][j] = (M[i][j]+1)/(total_trans_from_1+3)
            elif i == 2:
                M[i][j] = (M[i][j]+1)/(total_trans_from_2+3)
    return M


y_train = y_train.to_numpy()
m = get_transition_matrix(y_train)
for r in m: print(' '.join('{0:.5f}'.format(x) for x in r))

# get last element in sequence (in the dataset)
last_event = y_train[-1]
# print(last_event)
print("Next event in sequence of categories. Prediction of next item's category.")
print("Probability to transition to an event of type 'addtocart': ", m[last_event][0])
print("Probability to transition to an event of type 'transaction': ", m[last_event][1])
print("Probability to transition to an event of type 'view': ", m[last_event][2])


### TRANSITIONS GRAPH

# create a function that maps transition probability dataframe
# to markov edges and weights

def _get_markov_edges(Q):
    edges = {}
    for col in Q.columns:
        for idx in Q.index:
            edges[(idx, col)] = Q.loc[idx, col]
    return edges


def get_transitions_diagram(m):
    edges_wts = _get_markov_edges(pd.DataFrame(m))
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
    nx.drawing.nx_pydot.write_dot(G, 'retail_rocket_markov_chain.dot')
    nx.draw(G)
    plt.show()'''