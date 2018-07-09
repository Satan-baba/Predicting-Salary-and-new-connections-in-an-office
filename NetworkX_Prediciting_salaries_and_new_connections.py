import networkx as nx
import pandas as pd
import numpy as np
import pickle
P1_Graphs = pickle.load(open('A4_graphs','rb'))
P1_Graphs
for graph in P1_Graphs:
    print(graph)
    #using this data to find the type of graph and returning the type



G = nx.read_gpickle('email_prediction.txt')
print(nx.info(G))


# Salary Prediction

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
def salary_predictions():
    df = pd.DataFrame(index = G.nodes())
    man_sal = pd.Series(nx.get_node_attributes(G, 'ManagementSalary'))
    depart = pd.Series(nx.get_node_attributes(G, 'Department'))
    df['man_sal'] = man_sal
    df['degree'] = pd.Series(G.degree())
    df['clustering'] = pd.Series(nx.clustering(G))
    df['degree_centrality'] = pd.Series(nx.degree_centrality(G))
    df['closeness_centrality'] = pd.Series(nx.closeness_centrality(G, normalized = True))
    df['betweenness_centrality'] = pd.Series(nx.betweenness_centrality(G, normalized = True))
    df['pagerank'] = pd.Series(nx.pagerank(G))
    train = df.dropna()
    final_test = df[df['man_sal'].isnull() == True]
    final_test.drop(['man_sal'], axis = 1, inplace = True)
    y = train['man_sal']
    X = train.drop(['man_sal'], axis = 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)
    clf = RandomForestClassifier(n_estimators = 100, max_depth = 5, max_features = None, random_state = 0)
    ran = clf.fit(X, y)
    pred = ran.predict_proba(final_test)
    pred1 = [i[1] for i in pred]
    final_test['pred'] = pred1
    return final_test['pred']
#predicitons with random forest gives an roc auc score of 0.92 whereas svm and logistic regression give only about 0.79


#New Connections Prediction

future_connections = pd.read_csv('Future_Connections.csv', index_col=0, converters={0: eval})

def new_connections_predictions():
    future_connections['preferential_attachment'] = [i[2] for i in nx.preferential_attachment(G, future_connections.index)]
    future_connections['Common Neighbors'] = future_connections.index.map(lambda x: len(list(nx.common_neighbors(G, x[0], x[1]))))
    future_connections['resource_allocation'] = [i[2] for i in nx.resource_allocation_index(G, future_connections.index)]
    future_connections['jaccard'] = [i[2] for i in nx.jaccard_coefficient(G, future_connections.index)]
    final_test = future_connections[future_connections['Future Connection'].isnull() == True]
    train = future_connections.dropna()
    final_test.drop(['Future Connection'], axis = 1, inplace= True)
    X = train.drop(['Future Connection'], axis = 1)
    y = train['Future Connection']
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 3)
    clf = RandomForestClassifier(n_estimators = 100, max_depth = 5, max_features = None, random_state = 0)
    ran = clf.fit(X, y)
    pred = ran.predict_proba(final_test)
    pred1 = [i[1] for i in pred]
    final_test['pred'] = pred1
    
    return final_test['pred']


