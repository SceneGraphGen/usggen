from generate import *
from data import *


## Evaluation of Node-conditioned Edge prediction ################

def cond_edge_pred_accuracy(params, config, graph_data,
                            models, ind_to_predicates):
    labels=ind_to_predicates[list(np.arange(50))]
    labels = np.append(labels, 'no_edge')
    true_y_all = list()
    pred_y_all = list()
    for graph in graph_data:
        Xin, Fin = graph
        n = Xin.shape[0]
        Fin = np.where(Fin==0, params.no_edge_token, Fin)
        Fto_true = np.triu(Fin, +1)
        Ffrom_true = np.transpose(np.tril(Fin, -1))
        true_y = list(np.append(Fto_true[np.triu_indices(n, k = 1)], 
                                Ffrom_true[np.triu_indices(n, k = 1)],
                                axis=0).astype(int))
        X, Fto_pred, Ffrom_pred = generate_edges_given_nodes(params, config, Xin, models)
        pred_y = list(np.append(Fto_pred[np.triu_indices(n, k = 1)], 
                                Ffrom_pred[np.triu_indices(n, k = 1)],
                                axis=0))
        true_y_all.extend(true_y)
        pred_y_all.extend(pred_y)

    true_labels_all = list()
    for idx in true_y_all:
        if idx==51:
            true_labels_all.append('no_edge')
        else:
            true_labels_all.append(ind_to_predicates[idx-1])
    pred_labels_all = list()
    for idx in pred_y_all:
        if idx==51:
            pred_labels_all.append('no_edge')
        else:
            pred_labels_all.append(ind_to_predicates[idx-1])

    cm = confusion_matrix(true_labels_all, pred_labels_all,
                          labels=labels)
    df_cm = pd.DataFrame(cm/cm.sum(axis=1)[:, np.newaxis], index = labels,
                         columns = labels)
    sn.heatmap(df_cm, annot=False)
    plt.show()
    return cm



def edge_dist_given_node_pair(graphs):
    """
    Computes the probability distribution of edges (categorical) for all
    node pairs from the dataset
    """
    edge_dist = {}
    for graph in graphs:
        X, F = graph
        l = X.shape[0]
        for idx1 in range(l):
            for idx2 in range(idx1):
                # to edges
                key = (X[idx2], X[idx1])
                if key not in edge_dist:
                    edge_dist[key] = dict()
                edge = int(F[idx2, idx1])
                if edge not in edge_dist[key]:
                    edge_dist[key][edge] = 1
                else:
                    edge_dist[key][edge] += 1
                # from edges
                key = (X[idx1], X[idx2])
                if key not in edge_dist:
                    edge_dist[key] = dict()
                edge = int(F[idx1, idx2])
                if edge not in edge_dist[key]:
                    edge_dist[key][edge] = 1
                else:
                    edge_dist[key][edge] += 1
    for x1 in range(1, 151):
        for x2 in range(1, 151):
            # to edges
            key = (x2, x1)
            if key not in edge_dist:
                edge_dist[key] = dict()
            for idx in range(51):
                if idx not in edge_dist[key]:
                    edge_dist[key][idx] = 0
            edge_dist[key] = dict(sorted(edge_dist[key].items()))
    return edge_dist


# def fisher_similarity_test(gen_edge_dist, true_edge_dist):
#     fisher_sim_dict = dict()
#     for node_pair in gen_edge_dist:
#         m = np.array(list(gen_edge_dist[node_pair].values()), list(true_edge_dist[node_pair].values())).transpose()
#         res = stats.fisher_test(m)
#         fisher_sim_dict[node_pair] = res[0][0]
#     return fisher_sim_dict


def get_node_set(graphs):
    node_data = []
    for graph in graphs:
        X, _ = graph
        nodes = dict()
        for obj in X:
            obj = int(obj)-1
            if obj not in nodes:
                nodes[obj] = int(1)
            else:
                nodes[obj] += int(1)
        nodes = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[0], reverse=False)}
        obj_tup = (torch.Tensor(list(nodes.keys())).to(DEVICE), torch.Tensor(list(nodes.values())).to(DEVICE))
        node_data.append(obj_tup)
    return node_data

# --------------- MMD evaluation ----------------------------------#


def dirac_set_kernel(Xa, Xb):
    """
    Computes the kernel between two sets of objects (nodes)
    """
    na, ca = Xa
    na = na.long()
    ca = ca.float()
    nodes_a = torch.zeros(150).to(DEVICE)
    nodes_a[na] = 1
    counts_a = torch.zeros(150).to(DEVICE)
    counts_a[na] = ca
    nb, cb = Xb
    nb = nb.long()
    cb = cb.float()
    nodes_b = torch.zeros(150).to(DEVICE)
    nodes_b[nb] = 1
    counts_b = torch.zeros(150).to(DEVICE)
    counts_b[nb] = cb
    category_kernel = nodes_a*nodes_b                 
    count_kernel = 1/(1 + torch.abs(counts_a - counts_b))
    #print(category_kernel) 
    #print(count_kernel)
    count_kernel[counts_a==0] = 0
    count_kernel[counts_b==0] = 0
    kernel = torch.dot(category_kernel, count_kernel)#/(np.sqrt(na.shape[0]*nb.shape[0]))
    #kernel = torch.sum(category_kernel)/(na.shape[0]*nb.shape[0])
    return kernel

def dirac_set_kernel_test(Xa, Xb):
    Kab = dirac_set_kernel(Xa, Xb)
    Kaa = dirac_set_kernel(Xa, Xa)
    Kbb = dirac_set_kernel(Xb, Xb)
    return Kab/max(Kaa, Kbb)

def dirac_random_walk_kernel(Ga, Gb, walk_length=3):
    """
    Computes the random walk kernel of two input graphs for a given walk length.
    The node and edge kernels are 1 if categories match otherwise 0.
    """
    Xa, Fa = Ga
    Xb, Fb = Gb
    len_a = Xa.shape[0]
    len_b = Xb.shape[0]
    Xa_count = Counter(list(Xa))
    Xb_count = Counter(list(Xb))
    # init kernel table
    K = np.zeros((walk_length, len_a, len_b))
    # all combinations of node pair
    node_pairs = list(itertools.product(*[np.arange(len_a), np.arange(len_b)]))
    # iteratively estimate kernel for increasing path length
    for p in range(walk_length):
        for pair in node_pairs:
            r = pair[0]
            s = pair[1]
            if Xa[r]==Xb[s]:  # node match
                if p==0:
                    K[p, r, s] = 1/(Xa_count[Xa[r]]*Xb_count[Xb[s]])  # set init kernel
                else:
                    # get neighbors of r and s
                    Nr_lst = np.nonzero(Fa[r,:])[0]
                    Ns_lst = np.nonzero(Fb[s,:])[0]
                    # all combinations of node pairs in neighbors
                    N_node_pairs = list(itertools.product(*[Nr_lst, Ns_lst]))
                    # no neighbors of atleast one of r and s
                    if N_node_pairs==[]:
                        if Nr_lst==[] and Ns_lst==[]:
                            neighbor_sim = 1   # when both r and s has no neighbors
                        else:
                            neighbor_sim = 0.5 # when one of r and s has no neighbors
                    else:
                        # when both r and s have neighbors
                        neighbor_sim = 0
                        for N_pair in N_node_pairs:
                            Nr = N_pair[0]
                            Ns = N_pair[1]
                            if Fa[r, Nr] == Fb[s, Ns]:
                                neighbor_sim += K[p-1, Nr, Ns]
                    # update kernel for current RW of order p
                    K[p, r, s] = K[0, r, s]*neighbor_sim
    kernel = np.sum(K[walk_length-1])
    return kernel

def dirac_random_walk_kernel_test(Ga, Gb):
    Kab = dirac_random_walk_kernel(Ga, Gb)
    Kaa = dirac_random_walk_kernel(Ga, Ga)
    Kbb = dirac_random_walk_kernel(Gb, Gb)
    return Kab/max(Kaa, Kbb)

def compute_gram_matrix(graph_data, kernel):
    """
    Computes the NxN normalized gram matrix for the random walk kernel. 
    """
    num_graphs = len(graph_data)
    # init gram matrix
    gram_matrix = np.zeros((num_graphs, num_graphs))
    # get row column idx list
    pair_idx_lst = list(itertools.product(*[np.arange(num_graphs), np.arange(num_graphs)]))
    # compute gram matrix
    for pair_idx in pair_idx_lst:
        a, b = pair_idx
        gram_matrix[pair_idx] = kernel(graph_data[a], graph_data[b])
    # normalize gram matrix
    for pair_idx in pair_idx_lst:
        a, b = pair_idx
        if a!=b:
            gram_matrix[pair_idx] /= max(gram_matrix[a, a], gram_matrix[b, b])
    # remove diagonal elements
    for idx in range(num_graphs):
        gram_matrix[idx, idx] = 0
    return gram_matrix

def compute_mmd(graphs_gen, graphs_data, kernel):
    num_gen = len(graphs_gen)
    num_data = len(graphs_data)
    num_all = num_gen + num_data
    all_graphs = graphs_gen + graphs_data
    gram_matrix = compute_gram_matrix(all_graphs, kernel)
    total_xx = 1 if num_gen==1 else num_gen*(num_gen-1)
    mmd_xx = 1 if num_gen==1 else np.sum(gram_matrix[0:num_gen, 0:num_gen])/total_xx
    print(mmd_xx)
    total_yy = 1 if num_data==1 else num_data*(num_data-1)
    mmd_yy = 1 if num_data==1 else np.sum(gram_matrix[num_gen:num_all, num_gen:num_all])/total_yy
    print(mmd_yy)
    mmd_xy = np.sum(gram_matrix[num_gen:num_all, 0:num_gen])/(num_data*num_gen)
    print(mmd_xy)
    mmd = mmd_xx + mmd_yy - 2*mmd_xy
    return mmd
