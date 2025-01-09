import networkx as nx
import numpy as np
import logging
logger = logging.getLogger(__name__)

def graph_generator(params):
    node_no = params['node_no']
    edge_p = params['edge_p']
    vers = params['vers']

    g = nx.gnp_random_graph(node_no, edge_p, directed=True)
    unweighted_graph_name = f'digraph_{node_no:03}_{edge_p:03}_v{vers:03}_unw'
    logger.info(f'Graph created: {unweighted_graph_name}')
    return nx.node_link_data(g)

def convert_to_adj(graph):
    G = nx.node_link_graph(graph)
    return nx.adjacency_matrix(G).toarray()

### Expand a weighted digraph to eliminate vertices with out-dgree=0
def digraph_adj_expand(w_adj):
    exp_adj = w_adj.copy() #.toarray()
    for r in range(w_adj.shape[0]):
        r_sum = np.count_nonzero(w_adj[r])
        if r_sum == 0:
            # No outgoing links - create a loop
            exp_adj[r, r] = 1.0
    return exp_adj

def digraph_adj_weigh(unw_adj, params):
    w_adj = unw_adj.copy().astype(float)
    for r in range(unw_adj.shape[0]):
        r_sum = sum(unw_adj[r])
        r_nz = np.count_nonzero(unw_adj[r])
        if r_sum != 0.0:
            # Edges available - generate weights
            if params['method'] == 'rand':
                nz_weights = np.random.random(r_nz)
            else:
                nz_weights = np.array([num*1.0 for num in unw_adj[r] if num])
            nz_weights /= nz_weights.sum()
            w_no = 0
            for c in range(unw_adj.shape[1]):
                if unw_adj[r, c] > 0:
                    w_adj[r, c] = nz_weights[w_no]
                    w_no += 1
    return np.around(w_adj, params['num'])

def graph_gen(g_adj_weighed):
    g_new = nx.DiGraph(g_adj_weighed)
    return nx.node_link_data(g_new)