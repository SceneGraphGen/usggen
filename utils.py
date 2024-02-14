from imports import *

# ----------Load and pre-process graphs----------------------------
def item_to_index(lst, item):
    return [i for i, x in enumerate(lst) if x == item]

def clean_graph(obj_pair_id, relations):
    # multiple identical triplets are reduced to one.
    # multiple edges from one node to another is reduced to one
    obj_pair_lst = [tuple(row) for row in obj_pair_id]
    duplicate_id = [item_to_index(obj_pair_lst, x) for x in set(obj_pair_lst) 
                    if obj_pair_lst.count(x) > 1]
    id_to_delete = list()
    for dup_ids in duplicate_id:
        id_to_delete.extend(dup_ids[1:])
    
    obj_pair_id = np.delete(obj_pair_id, id_to_delete, axis=0)
    relations = np.delete(relations, id_to_delete, axis=0)
    
    return obj_pair_id, relations


def load_scene_graphs(graphs_file, info_file, num_graphs=None):
    # Read graph file
    roi_h5 = h5py.File(graphs_file, 'r')
    
    # Load contents from file
    all_relations = roi_h5['predicates'][:, 0]
    all_objects = roi_h5['labels'][:, 0]
    all_obj_pair_id = roi_h5['relationships']
    assert (all_obj_pair_id.shape[0] == all_relations.shape[0])
    
    im_to_first_rel = roi_h5['img_to_first_rel']
    im_to_last_rel = roi_h5['img_to_last_rel']
    assert (im_to_first_rel.shape[0] == im_to_last_rel.shape[0])
    
    all_boxes = roi_h5['boxes_512'][:]  # will index later
    assert np.all(all_boxes[:, :2] >= 0)  # sanity check
    assert np.all(all_boxes[:, 2:] > 0)  # no empty box
    # convert from xc, yc, w, h to x1, y1, x2, y2
    all_boxes[:, :2] = all_boxes[:, :2] - all_boxes[:, 2:] / 2
    all_boxes[:, 2:] = all_boxes[:, :2] + all_boxes[:, 2:]

    im_to_first_box = roi_h5['img_to_first_box']
    im_to_last_box = roi_h5['img_to_last_box']
    assert (im_to_first_box.shape[0] == im_to_last_box.shape[0])
    
    # segregate graphs in a list
    graph_list = list()
    if num_graphs == -1:
        num_graphs = len(im_to_last_rel)
        idx_list = np.arange(len(im_to_last_rel))
    else:
        idx_list = np.random.choice(np.arange(len(im_to_last_rel)), num_graphs,
                                    replace=False)
    
    for i in idx_list:     
        boxes_i = all_boxes[im_to_first_box[i]:im_to_last_box[i] + 1, :]
        gt_classes_i = all_objects[im_to_first_box[i]:im_to_last_box[i] + 1] - 1
        obj_pair_id =  all_obj_pair_id[im_to_first_rel[i]: im_to_last_rel[i] + 1, :]
        relations = all_relations[im_to_first_rel[i]: im_to_last_rel[i] + 1] - 1

        if relations.shape[0] != 0:
            obj_pair_id, relations = clean_graph(obj_pair_id, relations)
            obj_pair = np.column_stack((all_objects[obj_pair_id [:, 0]]-1,
                                        all_objects[obj_pair_id [:, 1]]-1))
            triplets = np.column_stack((obj_pair, relations))

            graph = {'obj_pair_id': obj_pair_id,
                     'triplets': triplets,
                     'objects': gt_classes_i,
                     'boxes': boxes_i,
                    }        
            
            # map objects to dict with index of occurence of that object
            obj_to_idx = dict( (x, item_to_index(gt_classes_i, x)) for i,x in enumerate(set(gt_classes_i)) )
    
            # map obj_ids of all triplets to those unique ids 
            triplet_obj_ids = np.unique(obj_pair_id)
            triplet_to_object_map = dict()
            for obj_id in triplet_obj_ids:
                obj = all_objects[obj_id]-1
                triplet_to_object_map[obj_id] = obj_to_idx[obj].pop(0)
            
            graph['triplet_to_object_map'] = triplet_to_object_map
            
            graph_list.append(graph)
        
    print('Total number of scene graphs: ', len(graph_list))
    
    return graph_list


def get_vocab(info_file):
    # Read graph vocabulary
    info = json.load(open(info_file, 'r'))
    class_to_ind = info['label_to_idx']
    predicate_to_ind = info['predicate_to_idx']

    ind_to_classes = np.array(sorted(class_to_ind, key=lambda k: class_to_ind[k]))
    ind_to_predicates = np.array(sorted(predicate_to_ind,
                                        key=lambda k: predicate_to_ind[k]))

    object_count = {k: v for k, v in sorted(info['object_count'].items(),
                                            key=lambda item: item[1], reverse=True)}
    relation_count = {k: v for k, v in sorted(info['predicate_count'].items(),
                                              key=lambda item: item[1], reverse=True)}
    
    print('Total number of classes: ', ind_to_classes.shape[0])
    print('Total number of predicates: ', ind_to_predicates.shape[0])
    
    return ind_to_classes, ind_to_predicates, object_count, relation_count


def get_X_A_F(graph):
    
    total_objects = list(graph['objects'])
    obj_pair_id = graph['obj_pair_id']
    triplet_to_object_map = graph['triplet_to_object_map']

    N = len(total_objects)
    A = np.zeros((N,N))
    F = np.zeros((N, N))
    for obj_pair, triplet in zip(obj_pair_id, graph['triplets']):
        A[triplet_to_object_map[obj_pair[0]],
          triplet_to_object_map[obj_pair[1]]] = 1
        F[triplet_to_object_map[obj_pair[0]],
          triplet_to_object_map[obj_pair[1]]] = triplet[2]+1
    
    X = graph['objects']+1
    #sA = sparse.csr_matrix(A)
    #sF = sparse.csr_matrix(F)

    return X, A, F
# --------------------------------------------------------------------------


# -------------- Visualizing Graphs ----------------------------------------
def viz_scene_graph(graph, ind_to_classes, ind_to_predicates):
    nodes = graph['objects']
    triplets=graph['triplets']
    object_pairs = graph['obj_pair_id']
    triplet_to_object_map = graph['triplet_to_object_map']

    sg = Digraph('sg')
    for idx, node in enumerate(nodes):
        sg.node(str(idx), ind_to_classes[node])
    
    for idx, triplet in enumerate(triplets):
        obj = object_pairs[idx, 0]
        subj = object_pairs[idx, 1]
        sg.edge(str(triplet_to_object_map[obj]),
        str(triplet_to_object_map[subj]),
        label= ind_to_predicates[triplet[2]])
        
    return sg
# --------------------------------------------------------------------------


# ---------------- PLOTTING ------------------------------------------------
def plot_histogram(lst, bins, title, density=True, scale=None, xtick_rotation=0, xlabel=None, ylabel=None, fit_curve=True):
    plt.rcParams["figure.figsize"] = (7,3)
    plt.rcParams.update({'font.size': 15})
    plt.title(title)
    n,x,_ = plt.hist(lst, bins=bins, density=density, histtype=u'step')
    if scale is not None:
        plt.yscale('log')
    if xlabel is not None:
        plt.xlabel(xlabel)
    if ylabel is not None:
	    plt.ylabel(ylabel)
    plt.xticks(rotation=xtick_rotation)
    if fit_curve:
        bin_centers = 0.5*(x[1:]+x[:-1])
        plt.plot(bin_centers,n)
    plt.show()


def plot_bar(path, lst1, lst2, title,  xtick_rotation=0, ymax=1, scale=None,xlabel=None, ylabel=None, ylen=4):
    plt.rcParams["figure.figsize"] = (20,ylen)
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({'figure.autolayout': True})
    plt.title(title)
    plt.bar(lst1, lst2, align='center')
    if scale is not None:
        plt.yscale('log')
    if xlabel is not None:
	    plt.xlabel(xlabel)
    if ylabel is not None:
	    plt.ylabel(ylabel)
    plt.ylim([0.00001, ymax])
    plt.xticks(rotation=xtick_rotation)
    plt.savefig(os.path.join(path, title))
    plt.show()
    return plt


def plot_corr_one_to_one(l, r, obj_idx, graph_list, ind_to_classes, ind_to_predicates):
    assert l==0 or l==1
    all_occ = dict()
    for graph in graph_list:
        for row in graph['triplets']:
            if row[l]==obj_idx:
                if row[r] not in all_occ:
                    all_occ[row[r]] = 1
                else:
                    all_occ[row[r]] += 1
    
    total = sum(list(all_occ.values()))
    if r==2:
        all_occ = {ind_to_predicates[k]: v/total for k, v in 
                   sorted(all_occ.items(), key=lambda item: item[1], reverse=True)}
    else:
        all_occ = {ind_to_classes[k]: v/total for k, v in 
                   sorted(all_occ.items(), key=lambda item: item[1], reverse=True)}
    if l==0:
        text = 'Object = '
    else:
        text = 'Subject = '
    print(text, ind_to_classes[obj_idx], obj_idx)
    plt.rcParams["figure.figsize"] = (20,3)
    plt.rcParams.update({'font.size': 15})
    x = list(all_occ.keys()) if len(list(all_occ.keys()))<50 else list(all_occ.keys())[:50]
    y = list(all_occ.values()) if len(list(all_occ.keys()))<50 else list(all_occ.values())[:50]
    plt.bar(x, y, align='center')
    plt.xticks(rotation=90)
    plt.show()

def sort_by_values_len(d, reverse=True):
    dict_len= {key: len(value) for key, value in d.items()}
    import operator
    sorted_key_list = sorted(dict_len.items(), key=operator.itemgetter(1), reverse=reverse)
    sorted_dict = [{item[0]: d[item [0]]} for item in sorted_key_list]
    return sorted_dict

def get_corr_dict_sorted(lst, ind_to_predicates, reverse=True):
    corr_dict = dict()
    for item in lst:
        if item not in corr_dict:
            corr_dict[item] = 1
        else:
            corr_dict[item] += 1

    total = sum(list(corr_dict.values()))
    corr_dict = {ind_to_predicates[k]: v/total for k, v in 
                   sorted(corr_dict.items(), key=lambda item: item[1], reverse=True)}
    return corr_dict

def plot_corr_two_to_one(graph_list, ind_to_classes, ind_to_predicates):
    l_tuple = (0,1)
    r = 2
    all_occ = dict()
    for graph in graph_list:
        for row in graph['triplets']:
            obj_pair = tuple(row[list(l_tuple)])
            if obj_pair not in all_occ:
                all_occ[obj_pair] = list()
            all_occ[obj_pair].append(row[r])
    
    all_occ_list = sort_by_values_len(all_occ)
    print('Most occuring object pairs:')
    for _ in range(3):
        pair_dict = all_occ_list[random.choice(range(20))]
        pair = list(pair_dict.keys())[0]
        pair_values = list(pair_dict.values())[0]
        pair_dict = get_corr_dict_sorted(pair_values, ind_to_predicates)
        print('object pair: ', tuple(ind_to_classes[list(pair)]))
        plt.rcParams["figure.figsize"] = (20,3)
        plt.rcParams.update({'font.size': 15})
        x = list(pair_dict.keys()) if len(list(pair_dict.keys()))<50 else list(pair_dict.keys())[:50]
        y = list(pair_dict.values()) if len(list(pair_dict.keys()))<50 else list(pair_dict.values())[:50]
        plt.bar(x, y, align='center')
        plt.xticks(rotation=90)
        plt.show()


    all_occ_list = sort_by_values_len(all_occ, reverse=False)
    print('Least occuring object pairs:')
    for _ in range(3):
        pair_dict = all_occ_list[random.choice(range(20))]
        pair = list(pair_dict.keys())[0]
        pair_values = list(pair_dict.values())[0]
        pair_dict = get_corr_dict_sorted(pair_values, ind_to_predicates)
        print('object pair: ', tuple(ind_to_classes[list(pair)]))
        plt.rcParams["figure.figsize"] = (20,3)
        plt.rcParams.update({'font.size': 15})
        x = list(pair_dict.keys()) if len(list(pair_dict.keys()))<50 else list(pair_dict.keys())[:50]
        y = list(pair_dict.values()) if len(list(pair_dict.keys()))<50 else list(pair_dict.values())[:50]
        plt.bar(x, y, align='center')
        plt.xticks(rotation=90)
        plt.show()


# ----------------------------------------------------------------------------------



# --------------- GRAPH PATTERNS ----------------------------------------------------
def get_degrees(A):
    indeg = np.sum(A, axis=1)
    outdeg = np.sum(A, axis=0)
    return indeg, outdeg

def get_degree_stats(graph_list):
    indegree = list()
    outdegree = list()
    for graph in graph_list:
        graph_in, graph_out = get_degrees(graph['A'])
        indegree.extend(graph_in)
        outdegree.extend(graph_out)
    return outdegree, indegree


def handle_zero_div(x, y):
    return x / (y + (y==0))


def directed_CCs_nodes(graph):
    A = graph['A']
    indeg, outdeg = get_degrees(A)

    bilateral = np.diag(A@A)
    denom_bi = indeg * outdeg - bilateral
    denom_in = indeg * (indeg - 1)
    denom_out = outdeg * (outdeg - 1)

    CC_cycle = handle_zero_div(np.diag(A@A@A), denom_bi)
    CC_middleman = handle_zero_div(np.diag(A@(A.T)@A), denom_bi)
    CC_in = handle_zero_div(np.diag((A.T)@A@A), denom_in)
    CC_out = handle_zero_div(np.diag(A@A@(A.T)), denom_out)
    A_tot = A + A.T
    denom_total = 2*(2*denom_bi + denom_in + denom_out)
    CC_total = handle_zero_div(np.diag(A_tot@A_tot@A_tot), denom_total)
    
    return CC_cycle, CC_middleman, CC_in, CC_out, CC_total


def CCs_graphs(graph_list):
    CC_cycle_list  = list()
    CC_middleman_list  = list()
    CC_in_list  = list()
    CC_out_list = list()
    CC_total_list = list()
    for graph in graph_list:
        CC_cycle, CC_middleman, CC_in, CC_out, CC_total = directed_CCs_nodes(graph)
        CC_cycle_list.append(np.mean(CC_cycle))
        CC_middleman_list.append(np.mean(CC_middleman))
        CC_in_list.append(np.mean(CC_in))
        CC_out_list.append(np.mean(CC_out))
        CC_total_list.append(np.mean(CC_total))
        
    return CC_cycle_list, CC_middleman_list, CC_in_list, CC_out_list, CC_total_list


def diameter_stats(graph_list):
    diameter_lst = list()
    eccentricity_lst = list()
    num_components_lst = list()
    for graph in graph_list:
        G=nx.Graph(graph['A'])
        num_components = 0
        for g in nx.connected_component_subgraphs(G):
            num_components += 1
            diameter_lst.append(nx.diameter(g))
            eccentricity_lst.extend(list(nx.eccentricity(g).values()))
        num_components_lst.append(num_components)
        
    return diameter_lst, eccentricity_lst, num_components_lst
# -------------------------------------------------------------------------------


# ----------------NODE and EGDE weights for CrossEntropyLoss--------------------
def normalize(d, norm):
    factor=norm/sum(d.values())
    return {k:v*factor for k,v in d.items()}


def get_node_freq(params, graphs_all):
    node_freq = {}
    for graph in graphs_all:
        X = graph[0]
        nodes, counts = np.unique(X, return_counts=True)
        for node, count in zip(nodes, counts):
            if node not in node_freq:
                node_freq[int(node)] = count
            else:
                node_freq[int(node)] += count

        if params.node_EOS_token+1 not in node_freq:
            node_freq[params.node_EOS_token+1] = 1
        else:
            node_freq[params.node_EOS_token+1] += 1
    
    return dict(sorted(node_freq.items()))


def get_edge_freq(params, graphs_all):
    edge_freq = {}
    for graph in graphs_all:
        F = graph[1]-1
        edges, counts = np.unique(F, return_counts=True)
        #print(F)
        #print(edges, counts)
        for edge, count in zip(edges, counts):
            if edge == -1:
                count -= F.shape[0]
                #print(count, F.shape[0])
            if edge not in edge_freq:
                edge_freq[int(edge)] = count
            else:
                edge_freq[int(edge)] += count                
        #print(edge_freq[-1])
    edge_freq[params.no_edge_token] = edge_freq.pop(-1)
    
    return dict(sorted(edge_freq.items()))


def inverse_freq_weights(params, node_freq, edge_freq):
    node_freq = {k:1.0/v for k,v in node_freq.items()}
    edge_freq = {k:1.0/v for k,v in edge_freq.items()}
    node_freq = normalize(node_freq, params.mlp_out_size)
    edge_freq = normalize(edge_freq, params.egru_output_size)
    
    node_weights = list(node_freq.values())
    print(len(list(node_freq.values())))
    edge_weights = list(edge_freq.values())
    assert len(node_weights) == params.mlp_out_size
    assert len(edge_weights) == params.egru_output_size
    
    return torch.Tensor(node_weights).to(DEVICE).float(), torch.Tensor(edge_weights).to(DEVICE).float()
    
    
def sqrt_inverse_freq_weights(params, node_freq, edge_freq):
    node_freq = {k:math.sqrt(1.0/v) for k,v in node_freq.items()}
    edge_freq = {k:math.sqrt(1.0/v) for k,v in edge_freq.items()}
    node_freq = normalize(node_freq, params.mlp_out_size)
    edge_freq = normalize(edge_freq, params.egru_output_size)
    
    node_weights = list(node_freq.values())
    edge_weights = list(edge_freq.values())
    assert len(node_weights) == params.mlp_out_size
    assert len(edge_weights) == params.egru_output_size
    
    return torch.Tensor(node_weights).to(DEVICE).float(), torch.Tensor(edge_weights).to(DEVICE).float() 




# ---------------------------------------------------

def remove_isolated_edges(graphs_all):
    new_graphs_all=list()
    for graph in graphs_all:
        X, F = graph
        connected_comp = list(get_connected_comp(X, F))
        connected_idx = np.array([i for idx in connected_comp for i in idx])
        X = X[connected_idx] # 0 to 149
        F = F[np.ix_(connected_idx, connected_idx)] # 1 to 50
        new_graphs_all.append((X, F))
        
    return new_graphs_all



def get_connected_comp(graph):
    X, F = graph
    graph = {}
    root = np.random.choice(np.arange(X.shape[0]))
    for i in range(X.shape[0]):
        edges = set(np.nonzero(F[:,i])[0])
        edges.update(set(np.nonzero(F[i,:])[0]))
        edges = list(edges)
        if edges != []:
            shuffle(edges)
        graph[i] = edges
    
    connected_comp=set()
    connected_comp = bfs_traversal(graph, root, connected_comp=set())
    
    return connected_comp



def bfs_traversal(graph, root, connected_comp=set()):
    #print('start bfs: ', graph, root)
    order = list()
    visited, queue = set(), collections.deque([root])
    while queue:
        vertex = queue.popleft()
        order.append(vertex)
        visited.add(vertex)
        #print('visited: ', visited)
        queue.extend(n for n in graph[vertex] if n not in visited and n not in queue)
        #print('queue: ', queue)
    
    if len(order)>1:
        connected_comp.add(tuple(order))
    #print('Order: ', order)
    #print()
    remaining = [node for node in graph if node not in order]
    if remaining!=[]:
        graph = {k:v for k,v in graph.items() if k in remaining}
        shuffle(remaining)
        root = remaining[0]
        connected_comp = bfs_traversal(graph, root, connected_comp=connected_comp)
        
    return connected_comp





def get_img_num_per_cls(cifar_version, imb_factor=None):
    """
    Get a list of image numbers for each class, given cifar version
    Num of imgs follows emponential distribution
    img max: 5000 / 500 * e^(-lambda * 0);
    img min: 5000 / 500 * e^(-lambda * int(cifar_version - 1))
    exp(-lambda * (int(cifar_version) - 1)) = img_max / img_min
    args:
      cifar_version: str, '10', '100', '20'
      imb_factor: float, imbalance factor: img_min/img_max,
        None if geting default cifar data number
    output:
      img_num_per_cls: a list of number of images per class
    """
    cls_num = int(cifar_version)
    img_max = img_num(cifar_version)
    if imb_factor is None:
        return [img_max] * cls_num
    img_num_per_cls = []
    for cls_idx in range(cls_num):
        num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
        img_num_per_cls.append(int(num))
    return img_num_per_cls
