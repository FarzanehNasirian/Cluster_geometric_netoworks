import networkx as nx
import pandas as pd
import numpy as np
import umap
from node2vec import Node2Vec
from sklearn import mixture
import random
from tqdm import tqdm

def GMM(G, k_min, k_max):
    
    #convert graph-tool graph to networkx
    g_nx = nx.from_pandas_edgelist(pd.DataFrame([e for e in G.edges()], columns=['source', 'target']))

    #embed the graph
    node2vec = Node2Vec(g_nx, dimensions=128, walk_length=80, num_walks=10, workers=1)
    model = node2vec.fit(window=10, min_count=1)
    
        #feature vectors
    X = [model.wv[str(v)] for v in g_nx.nodes()]
    X = umap.UMAP(random_state=42, n_components=3).fit(X)

    cord_map = {}
    c = 0
    for v in g_nx.nodes():
        cord_map[v] = [X.embedding_.T[0][c], X.embedding_.T[1][c], X.embedding_.T[2][c]]
        c += 1

    mapper = [[cord_map[item][0], cord_map[item][1], cord_map[item][2]] for item in g_nx.nodes()]         
    
    #tune number of cluster
    bic_scores = []
    print('tuning the number of clusters ...')
    for k in tqdm(range(k_min, k_max)):
        gmm = (mixture.GaussianMixture(n_components=k, random_state=0, covariance_type='full', n_init=10)
                    .fit(mapper))
        bic_scores.append([k, gmm.bic(np.array(mapper))])
    bic_scores = pd.DataFrame(bic_scores, columns=['k', 'bic'])
    k_tuned = bic_scores[bic_scores.bic == bic_scores.bic.min()]['k'].tolist()[0]
    
    gmm = (mixture.GaussianMixture(n_components=k_tuned, random_state=0, covariance_type='full', n_init=10)
            .fit(mapper))

    cdf = pd.DataFrame(zip(gmm.predict(mapper).tolist(), [v for v in g_nx.nodes()]),
                   columns=['cluster', 'node']).explode('node')
    
    r = lambda: random.randint(0,255)
    color_dict = {i:'#%02X%02X%02X' % (r(), r(), r()) for i in range(k_tuned)}
    
    cdf['color'] = cdf.cluster.apply(lambda x: color_dict[x])
    
    return(cdf)