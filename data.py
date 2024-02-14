from imports import *

IND_TO_CLASSES = ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book',
                 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock',
                 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence',
                 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat',
                 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter',
                 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw',
                 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing',
                 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski',
                 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower',
                 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield',
                 'wing', 'wire', 'woman', 'zebra']

MOTIF_OBJECT_RANK_DICT = {
#PLACE
'room': 1, 'tower': 1, 'hill': 1, 'house': 1, 'beach': 1, 'street': 1, 'building': 1, 'mountain': 1, 'sidewalk': 1, 'track': 1, 'toilet': 1,
#VEHICLE
'airplane': 2, 'vehicle': 2, 'train': 2, 'bus': 2, 'bike': 2, 'plane': 2, 'truck': 2, 'motorcycle': 2, 'boat': 2, 'car': 2,
#PEOPLE_ANIMAL_PLANT
'woman': 3, 'lady': 3, 'child': 3, 'guy': 3, 'girl': 3,  'men': 3, 'man': 3, 'people': 3, 'person': 3, 'boy': 3, 'kid': 3, 'skier': 3,
'player': 3, 'animal': 3, 'bear': 3, 'bird': 3, 'zebra': 3, 'cow': 3, 'cat': 3, 'dog': 3, 'elephant': 3, 'giraffe': 3, 'sheep': 3,
'horse': 3, 'tree': 3, 'plant': 3,
#FURNITURE
'basket': 4, 'box': 4, 'cabinet': 4, 'chair': 4, 'clock': 4, 'counter': 4, 'curtain': 4, 'desk': 4, 'door': 4, 'seat': 4, 'shelf': 4,
'fence': 4, 'board': 4, 'lamp': 4, 'vase': 4, 'sink': 4, 'bench': 4, 'bed': 4, 'stand': 4, 'table': 4, 'drawer': 4, 'pillow': 4, 
'light': 4, 'tile': 4, 'railing': 4,  'roof': 4, 'glass': 4,
#UTENSIL
'bottle': 5, 'bowl': 5, 'cup': 5, 'plate': 5, 'fork': 5, 'pot': 5,
#FOOD
'banana': 6, 'pizza': 6, 'vegetable': 6, 'orange': 6, 'food': 6, 'fruit': 6, 
#OUTDOOR_VEHICLE_PARTOF
'wheel': 7, 'window': 7, 'windshield': 7, 'handle': 7, 'engine': 7, 'tire': 7, 'pole': 7, 'post': 7, 'snow': 7, 'wave': 7, 'wire': 7,
'flag': 7, 'kite': 7, 'rock': 7, 'letter': 7,'logo': 7, 'paper': 7, 'number': 7, 'screen': 7, 'sign': 7,
#PEOPLE_ANIMAL_PLANT_PARTOF
'eye': 8, 'face': 8, 'ear': 8, 'hair': 8, 'hand': 8, 'head': 8, 'arm': 8, 'leg': 8, 'wing': 8, 'mouth': 8, 'neck': 8, 'nose': 8, 'paw': 8,
'branch': 8, 'finger': 8, 'flower': 8, 'leaf': 8, 'trunk': 8, 'tail': 8, 'head': 8,
#CLOTH_ACCESSORY 
'hat': 9, 'cap': 9, 'coat': 9, 'helmet': 9, 'jacket': 9, 'jean': 9, 'glove': 9, 'shirt': 9, 'shoe': 9, 'short': 9, 'tie': 9, 'pant': 9, 'boot': 9,
'sock': 9, 'skateboard': 9, 'ski': 9, 'sneaker': 9, 'surfboard': 9, 'racket': 9, 'bag': 9, 'book': 9, 'laptop': 9, 'phone': 9, 'towel': 9, 'umbrella': 9
}

MOTIF_OBJECT_RANK_DICT2 = {
#PLACE_VEHICLE
'room': 1, 'tower': 1, 'hill': 1, 'house': 1, 'beach': 1, 'street': 1, 'building': 1, 'mountain': 1, 'sidewalk': 1, 'track': 1, 'toilet': 1,
'airplane': 1, 'vehicle': 1, 'train': 1, 'bus': 1, 'bike': 1, 'plane': 1, 'truck': 1, 'motorcycle': 1, 'boat': 1, 'car': 1,
#PEOPLE_ANIMAL_PLANT_FURNITURE
'woman': 2, 'lady': 2, 'child': 2, 'guy': 2, 'girl': 2,  'men': 2, 'man': 2, 'people': 2, 'person': 2, 'boy': 2, 'kid': 2, 'skier': 2,
'player': 2, 'animal': 2, 'bear': 2, 'bird': 2, 'zebra': 2, 'cow': 2, 'cat': 2, 'dog': 2, 'elephant': 2, 'giraffe': 2, 'sheep': 2,
'horse': 2, 'tree': 2, 'plant': 2,
'basket': 2, 'box': 2, 'cabinet': 2, 'chair': 2, 'clock': 2, 'counter': 2, 'curtain': 2, 'desk': 2, 'door': 2, 'seat': 2, 'shelf': 2,
'fence': 2, 'board': 2, 'lamp': 2, 'vase': 2, 'sink':2, 'bench': 2, 'bed': 2, 'stand': 2, 'table': 2, 'drawer': 2, 'pillow': 2, 
'light': 2, 'tile': 2, 'railing': 2,  'roof': 2, 'glass': 2, 'bottle': 2, 'bowl': 2, 'cup': 2, 'plate': 2, 'fork': 2, 'pot': 2,
'banana': 2, 'pizza': 2, 'vegetable': 2, 'orange': 2, 'food': 2, 'fruit': 2,
#PARTOF PEOPLE_ANIMAL_PLANT_FURNITURE_VEHICLE
'wheel': 3, 'window': 3, 'windshield': 3, 'handle': 3, 'engine': 3, 'tire': 3, 'pole': 3, 'post': 3, 'snow': 3, 'wave': 3, 'wire': 3,
'flag': 3, 'kite': 3, 'rock': 3, 'letter': 3,'logo': 3, 'paper': 3, 'number': 3, 'screen': 3, 'sign': 3,
'eye': 3, 'face': 3, 'ear': 3, 'hair': 3, 'hand': 3, 'head': 3, 'arm': 3, 'leg': 3, 'wing': 3, 'mouth': 3, 'neck': 3, 'nose': 3, 'paw': 3,
'branch': 3, 'finger': 3, 'flower': 3, 'leaf': 3, 'trunk': 3, 'tail': 3, 'head': 3,
'hat': 3, 'cap': 3, 'coat': 3, 'helmet': 3, 'jacket': 3, 'jean': 3, 'glove': 3, 'shirt': 3, 'shoe': 3, 'short': 3, 'tie': 3, 'pant': 3, 'boot': 3,
'sock': 3, 'skateboard': 3, 'ski': 3, 'sneaker': 3, 'surfboard': 3, 'racket': 3, 'bag': 3, 'book': 3, 'laptop': 3, 'phone': 3, 'towel': 3, 'umbrella': 3
}

def motif_based_ordered(X, F, ind_to_classes):
    obj_rank = [[idx, MOTIF_OBJECT_RANK_DICT2[ind_to_classes[o-1]]] for idx, o in enumerate(X)]
    sorted_pair = np.array(sorted(obj_rank, key = lambda x: x[1]))
    order_idx = [int(i) for i in sorted_pair[:,0]]
    X = X[order_idx] # 0 to 149
    F = F[np.ix_(order_idx, order_idx)] # 1 to 50
    return X, F


def random_ordered(X, F):
    order_idx = np.arange(X.shape[0])
    shuffle(order_idx)
    X = X[order_idx] # 0 to 149
    F = F[np.ix_(order_idx, order_idx)] # 1 to 50
    return X, F


def predefined_ordered(X, F):
    # X, F = random_ordered(X, F)
    obj_idx_pair = [[idx, obj] for idx, obj in enumerate(X)]
    sorted_pair = np.array(sorted(obj_idx_pair, key = lambda x: x[1]))
    order_idx = [int(i) for i in sorted_pair[:,0]]
    X = X[order_idx] # 0 to 149
    F = F[np.ix_(order_idx, order_idx)] # 1 to 50
    return X, F



def bfs_ordered(X, F, root):
    graph = {}
    for i in range(X.shape[0]):
        edges = set(np.nonzero(F[:,i])[0])
        edges.update(set(np.nonzero(F[i,:])[0]))
        edges = list(edges)
        if edges != []:
            shuffle(edges)
        graph[i] = edges
    
    order_idx = np.array(breadth_first_search(graph, root))
    X = X[order_idx] # 0 to 149
    F = F[np.ix_(order_idx, order_idx)] # 1 to 50
    
    return X, F



def get_edge_to_noedge_ratio(F):
    n = F.shape[0]
    num_edges = np.count_nonzero(F)
    num_noedges = n*n - n - num_edges
    if num_noedges==0:
        return 1
    else:
        return num_edges/num_noedges


def breadth_first_search(graph, root):
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
    
    #print('Order: ', order)
    #print()
    remaining = [node for node in graph if node not in order]
    if remaining!=[]:
        graph = {k:v for k,v in graph.items() if k in remaining}
        shuffle(remaining)
        root = remaining[0]
        order.extend(breadth_first_search(graph, root))
    return order



def get_dist_connection_range(graphs_all, threshold, iterations):
    connection_range = list()
    outlier_graphs = set()
    for it in range(iterations):
        print('Iteration: ', it)
        for idx, graph in enumerate(graphs_all):
            X, F = graph
            X, F = bfs_ordered(X, F)
            Fto = np.triu(F, +1)
            Ffrom = np.transpose(np.tril(F, -1))
            for i in range(X.shape[0]):
                edge_idx = list(np.nonzero(Fto[:, i])[0])
                # print(edge_idx)
                if edge_idx != []:
                    dist = i-1-edge_idx[0]
                    if dist != 0:
                        connection_range.append(dist)
                    if dist>threshold:
                        # print(F, F.shape)
                        outlier_graphs.add(idx)
                        # print('Fto', idx, i, Fto[:,i], edge_idx)
                
                edge_idx = list(np.nonzero(Ffrom[:,i])[0])
                # print(edge_idx)
                if edge_idx != []:
                    dist = i-1-edge_idx[0]
                    if dist != 0:
                        connection_range.append(dist)
                    if dist>threshold:
                        # print(F, F.shape)
                        outlier_graphs.add(idx)
                        # print('Ffrom', idx, i, Ffrom[:,i], edge_idx)
                        
    print(max(connection_range))
    #plt.hist(connection_range)
    return connection_range, outlier_graphs
    
    



class Graph_sequence_sampler(Dataset):
    """
    Create a custom Dataset class for Dataloader
    """
    def __init__(self, graphs_all, params, ordering):
        print('Initializing dataloader with ', ordering, ' ordering')
        self.n = params.max_num_node
        self.ordering = ordering

        self.X_all = []
        self.F_all = []
        self.len_all = []
        for graph in graphs_all:
            X = graph[0]
            F = graph[1]
            if X.shape[0]<=self.n and X.shape[0]>=params.min_num_node:
                self.X_all.append(X)
                self.F_all.append(F)
                self.len_all.append(X.shape[0])
        
        self.node_EOS_token = params.node_EOS_token
        self.edge_SOS_token = params.edge_SOS_token
        self.no_edge_token = params.no_edge_token
            
    def __len__(self):
        return len(self.X_all)
    
    def __getitem__(self, idx):
        """
        Inputs:
        X can be 0 for padding, 1 to 150 for each object category
        F can be 0 for padding, 1 to 50 for each edge(relatinship) category, 51 for no_edge, 52 edge_SOS_token
        Outputs:
        X can be -1 for padding, 0 to 149 for each object category, 150 for node_EOS_token
        F can be -1 for padding, 0 to 49 for each category, 50 for no_edge
        """
        # get sample
        X = self.X_all[idx].copy()
        F = self.F_all[idx].copy() 
        len_item = self.len_all[idx]
        
        # order the graph to a sequence to get permuted X and F
        if self.ordering=='random':
            X, F = random_ordered(X, F)
        elif self.ordering == 'predefined':
            X, F = predefined_ordered(X, F)
        elif self.ordering == 'none':
            pass
        elif self.ordering=='bfs':
            root =  random.choice(np.arange(X.shape[0]))
            X, F = bfs_ordered(X, F, root)
        elif self.ordering=='3dbfs':
            root = list(X).index(32)
            X, F = bfs_ordered(X, F, root)
        elif self.ordering=='motif_based':
            X, F = motif_based_ordered(X, F, IND_TO_CLASSES)
        else:
            raise print('Ordering not understood')

        ##############################
        # num_FG_edges = np.count_nonzero(F)
        # get 'to' and 'from' edge matrix
        F = np.where(F==0, self.no_edge_token, F)
        Fto = np.triu(F, +1)
        Ffrom = np.transpose(np.tril(F, -1))
        
        # ------INPUTS OF THE MODEL------------------------
        # GRU_graph
        Xin_ggru = np.zeros(self.n)
        Xin_ggru[0:len_item] = X
        
        Fto_in_ggru = np.zeros((self.n, self.n))
        Fto_in_ggru[0, 0:len_item] = self.edge_SOS_token*np.ones(len_item)
        Fto_in_ggru[1:len_item, 0:len_item] = Fto[0:len_item-1, :]
        Fto_in_ggru = np.transpose(Fto_in_ggru)[:, 1:]
        
        Ffrom_in_ggru = np.zeros((self.n, self.n))
        Ffrom_in_ggru[0, 0:len_item] = self.edge_SOS_token*np.ones(len_item)
        Ffrom_in_ggru[1:len_item, 0:len_item] = Ffrom[0:len_item-1, :]
        Ffrom_in_ggru = np.transpose(Ffrom_in_ggru)[:, 1:]
        
        # GRU_edge
        Xseq1 = np.array([X[0:len_item-1],]*(len_item-1)).transpose()
        Xseq1 = np.triu(Xseq1)
        Xin_egru1 = np.zeros((self.n-1, self.n-1))
        Xin_egru1[0:len_item-1, 0:len_item-1] = Xseq1
        Xin_egru1[0:len_item-1, 0:len_item-1] = Xseq1
        Xin_egru1 = np.transpose(Xin_egru1)
        
        Xseq2 = np.array([X[1:len_item],]*(len_item-1))
        Xseq2 = np.triu(Xseq2)
        Xin_egru2 = np.zeros((self.n-1, self.n-1))
        Xin_egru2[0:len_item-1, 0:len_item-1] = Xseq2
        Xin_egru2 = np.transpose(Xin_egru2)
        
        Fto_in_egru = np.zeros((self.n-1, self.n-1))
        Fto_in_egru[0,0:len_item-1] = self.edge_SOS_token*np.ones(len_item-1)
        Fto_in_egru[1:len_item-1, 0:len_item-1] = np.triu(Fto, +2)[0:len_item-2, 1:len_item]
        Fto_in_egru = np.transpose(Fto_in_egru)
        
        Ffrom_in_egru = np.zeros((self.n-1, self.n-1))
        Ffrom_in_egru[0,0:len_item-1] = self.edge_SOS_token*np.ones(len_item-1)
        Ffrom_in_egru[1:len_item-1, 0:len_item-1] = np.triu(Ffrom, +2)[0:len_item-2, 1:len_item]
        Ffrom_in_egru = np.transpose(Ffrom_in_egru)

        Fto_in_egru_shifted = np.zeros((self.n-1, self.n-1))
        Fto_in_egru_shifted[0:len_item-1, 0:len_item-1] = Fto[0:len_item-1, 1:len_item]
        Fto_in_egru_shifted = np.transpose(Fto_in_egru_shifted)
        
        # ---------OUTPUTS OF THE MODEL------------------------
        # MLP_node
        Xout_mlp = -1*np.ones(self.n)
        Xout_mlp[0:len_item-1] = X[1:]-1
        Xout_mlp[len_item-1] = self.node_EOS_token

        Fto_out_egru = np.zeros((self.n-1, self.n-1))
        Fto_out_egru[0:len_item-1, 0:len_item-1] = Fto[0:len_item-1, 1:len_item]
        Fto_out_egru = np.transpose(Fto_out_egru)-1
    
        Ffrom_out_egru = np.zeros((self.n-1, self.n-1))
        Ffrom_out_egru[0:len_item-1, 0:len_item-1] = Ffrom[0:len_item-1, 1:len_item]
        Ffrom_out_egru = np.transpose(Ffrom_out_egru)-1
        
        # Ffrom_out_FG_mask = np.where(Ffrom_out_egru==self.no_edge_token-1, 0, Ffrom_out_egru)
        # Ffrom_out_FG_mask = np.where(Ffrom_out_FG_mask!=0, 1, Ffrom_out_FG_mask)
        
        # Fto_out_FG_mask = np.where(Fto_out_egru==self.no_edge_token-1, 0, Fto_out_egru)
        # Fto_out_FG_mask = np.where(Fto_out_FG_mask!=0, 1, Fto_out_FG_mask)

        return {
            # inputs
            'Xin_ggru': Xin_ggru,
            'Fto_in_ggru': Fto_in_ggru,
            'Ffrom_in_ggru': Ffrom_in_ggru,
            'Xin_egru1': Xin_egru1,
            'Xin_egru2': Xin_egru2,
            'Fto_in_egru': Fto_in_egru,
            'Ffrom_in_egru': Ffrom_in_egru,
            'Fto_in_egru_shifted': Fto_in_egru_shifted,
            # outputs
            'Xout_mlp': Xout_mlp,
            'Fto_out_egru': Fto_out_egru,
            'Ffrom_out_egru': Ffrom_out_egru,
            # sequence lengths
            'len': len_item,
            'num_edges': 0.5*(len_item-1)*len_item,
            #'num_FG_edges': num_FG_edges,
            #'edge_to_noedge_ratio': edge_to_noedge_ratio
            #'Ffrom_out_FG_mask': Ffrom_out_FG_mask,
            #'Fto_out_FG_mask': Fto_out_FG_mask
            }



def shorten_matrix_bfs(mat, n, max_edge_len):

    assert mat.shape[0]==n
    new_mat = np.zeros((max_edge_len, n))
    new_mat[:, 0:max_edge_len] = mat[0:max_edge_len, 0:max_edge_len]
    for i in range(max_edge_len, n):
        new_mat[:,i] = mat[i-max_edge_len+1:i+1,i]
    
    return new_mat
    
    

class Graph_sequence_sampler_bfs(Dataset):
    """
    Create a custom Dataset class for Dataloader
    """
    def __init__(self, graphs_all, params, ordering):
        print('Initializing dataloader with bfs ordering')
        self.X_all = []
        self.F_all = []
        self.len_all = []
        
        self.n = params.max_num_node
        self.max_edge_len = params.max_edge_len
        
        for graph in graphs_all:
            X = graph[0]
            F = graph[1]
            if X.shape[0]<=self.n:
                self.X_all.append(X)
                self.F_all.append(F)
                self.len_all.append(X.shape[0])
        
        self.node_EOS_token = params.node_EOS_token
        self.edge_SOS_token = params.edge_SOS_token
        self.no_edge_token = params.no_edge_token
        
    def __len__(self):
        return len(self.X_all)
    
    def __getitem__(self, idx):
        """
        Inputs:
        X can be 0 for padding, 1 to 150 for each object category
        F can be 0 for padding, 1 to 50 for each edge(relatinship) category, 51 for no_edge, 52 edge_SOS_token
        Outputs:
        X can be -1 for padding, 0 to 149 for each object category, 150 for node_EOS_token
        F can be -1 for padding, 0 to 49 for each category, 50 for no_edge
        """
        # get sample
        X = self.X_all[idx].copy()
        F = self.F_all[idx].copy() 
        len_item = X.shape[0]
        # order the graph to a sequence to get permuted X and F
        X, F = bfs_ordered(X, F)
        
        #print(X)
        #print(F)
        # get 'to' and 'from' edge matrix
        F = np.where(F==0, self.no_edge_token, F)
        Fto = np.triu(F, +1)
        Ffrom = np.transpose(np.tril(F, -1))
        #print(Fto)
        #print(Ffrom)
        # ------INPUTS OF THE MODEL------------------------
        # GRU_graph
        Xin_ggru = np.zeros(self.n)
        Xin_ggru[0:len_item] = X
        #print('Xin_ggru', Xin_ggru.shape, Xin_ggru[0:12])
        Fto_in_ggru = np.zeros((self.n, self.n))
        Fto_in_ggru[1:len_item, 0:len_item] = Fto[0:len_item-1, :]
        Fto_in_ggru = shorten_matrix_bfs(Fto_in_ggru, self.n, self.max_edge_len)
        Fto_in_ggru[0, 0:len_item] = self.edge_SOS_token*np.ones(len_item)
        Fto_in_ggru = np.transpose(Fto_in_ggru)[:, 1:]
        #print('Fto_in_ggru', Fto_in_ggru.shape, Fto_in_ggru[0:12, 0:12])
        Ffrom_in_ggru = np.zeros((self.n, self.n))
        Ffrom_in_ggru[1:len_item, 0:len_item] = Ffrom[0:len_item-1, :]
        Ffrom_in_ggru = shorten_matrix_bfs(Ffrom_in_ggru, self.n, self.max_edge_len)
        Ffrom_in_ggru[0, 0:len_item] = self.edge_SOS_token*np.ones(len_item)
        Ffrom_in_ggru = np.transpose(Ffrom_in_ggru)[:, 1:]
        #print('Ffrom_in_ggru', Ffrom_in_ggru.shape, Ffrom_in_ggru[0:12, 0:12])
        # GRU_edge
        # Xin_egru_hidden = np.zeros(self.n-1)
        # Xin_egru_hidden[0:len_item-1] = X[1:]
        # Xin_egru_hidden[len_item-1] = self.node_EOS_token
        # print('Xin_egru_hidden', Xin_egru_hidden.shape, Xin_egru_hidden[0:12])
        Xseq1 = np.array([X[0:len_item-1],]*(len_item-1)).transpose()
        Xseq1 = np.triu(Xseq1)
        Xin_egru1 = np.zeros((self.n-1, self.n-1))
        Xin_egru1[0:len_item-1, 0:len_item-1] = Xseq1
        Xin_egru1 = shorten_matrix_bfs(Xin_egru1, self.n-1, self.max_edge_len)
        Xin_egru1 = np.transpose(Xin_egru1)
        Xseq2 = np.array([X[1:len_item],]*(len_item-1))
        Xseq2 = np.triu(Xseq2)
        Xin_egru2 = np.zeros((self.n-1, self.n-1))
        Xin_egru2[0:len_item-1, 0:len_item-1] = Xseq2
        Xin_egru2 = shorten_matrix_bfs(Xin_egru2, self.n-1, self.max_edge_len)
        Xin_egru2 = np.transpose(Xin_egru2)
        #print('Xin_egru1', Xin_egru1.shape, Xin_egru1[0:12, 0:12])
        #print('Xin_egru2', Xin_egru2.shape, Xin_egru2[0:12, 0:12])
        Fto_in_egru = np.zeros((self.n-1, self.n-1))
        Fto_in_egru[0,0:len_item-1] = self.edge_SOS_token*np.ones(len_item-1)
        Fto_in_egru[1:len_item-1, 0:len_item-1] = np.triu(Fto, +2)[0:len_item-2, 1:len_item]
        Fto_in_egru = shorten_matrix_bfs(Fto_in_egru, self.n-1, self.max_edge_len)
        Fto_in_egru = np.transpose(Fto_in_egru)
        #print('Fto_in_egru', Fto_in_egru.shape, Fto_in_egru[0:12, 0:12])
        Ffrom_in_egru = np.zeros((self.n-1, self.n-1))
        Ffrom_in_egru[0,0:len_item-1] = self.edge_SOS_token*np.ones(len_item-1)
        Ffrom_in_egru[1:len_item-1, 0:len_item-1] = np.triu(Ffrom, +2)[0:len_item-2, 1:len_item]
        Ffrom_in_egru = shorten_matrix_bfs(Ffrom_in_egru, self.n-1, self.max_edge_len)
        Ffrom_in_egru = np.transpose(Ffrom_in_egru)
        #print('Ffrom_in_egru', Ffrom_in_egru.shape, Ffrom_in_egru[0:12, 0:12])
        
        # ---------OUTPUTS OF THE MODEL------------------------
        # get 'to' and 'from' edge matrix
        #Fout = F
        #Fout[Fout>0] -= 1 # 0 to 49
        #Fto = np.triu(Fout, +1)
        #Ffrom = np.transpose(np.tril(Fout, -1))
        # MLP_node
        #Xout_mlp = -1*np.ones(self.n)
        #Xout_mlp[0:len_item-1] = X[1:] - 1
        #Xout_mlp[len_item-1] = self.node_EOS_token
        #print('Xout_mlp', Xout_mlp.shape, Xout_mlp[0:12])
        # GRU_edge
        #Fto += np.tril(np.array([-1*np.ones(len_item),]*len_item), 0)
        #Ffrom += np.tril(np.array([-1*np.ones(len_item),]*len_item), 0)
        #Fto_out_egru = -1*np.ones((self.n-1, self.n-1))
        #Ffrom_out_egru = -1*np.ones((self.n-1, self.n-1))
        Fto_out_egru = np.zeros((self.n-1, self.n-1))
        Fto_out_egru[0:len_item-1, 0:len_item-1] = Fto[0:len_item-1, 1:len_item]
        Fto_out_egru = shorten_matrix_bfs(Fto_out_egru, self.n-1, self.max_edge_len)
        Fto_out_egru = np.transpose(Fto_out_egru) - 1
        #print('Fto_out_egru', Fto_out_egru.shape, Fto_out_egru[0:12, 0:12])
        
        Ffrom_out_egru = np.zeros((self.n-1, self.n-1))
        Ffrom_out_egru[0:len_item-1, 0:len_item-1] = Ffrom[0:len_item-1, 1:len_item]
        Ffrom_out_egru = shorten_matrix_bfs(Ffrom_out_egru, self.n-1, self.max_edge_len)
        Ffrom_out_egru = np.transpose(Ffrom_out_egru) - 1
        #print('Ffrom_out_egru', Ffrom_out_egru.shape, Ffrom_out_egru[0:12, 0:12])

        if len_item-1<=self.max_edge_len:
            num_edges = 0.5*(len_item-1)*len_item
        else:
            num_edges = 0.5*(self.max_edge_len+1)*self.max_edge_len + (len_item-1-self.max_edge_len)*self.max_edge_len
            
            
        return {
            # inputs
            'Xin_ggru': Xin_ggru,
            'Fto_in_ggru': Fto_in_ggru,
            'Ffrom_in_ggru': Ffrom_in_ggru,
            #'Xin_egru_hidden': Xin_egru_hidden,
            'Xin_egru1': Xin_egru1,
            'Xin_egru2': Xin_egru2,
            'Fto_in_egru': Fto_in_egru,
            'Ffrom_in_egru': Ffrom_in_egru,
            # outputs
            #'Xout_mlp': Xout_mlp,
            'Fto_out_egru': Fto_out_egru,
            'Ffrom_out_egru': Ffrom_out_egru,
            # sequence lengths
            'len': len_item,
            'num_edges': num_edges
            }
    


# split train and validation data for each epoch
# validate on a randomly uniformly drawn samples from training data (without replacement)
def train_validate_split(params, graphs_train):
    num_train_samples = len(graphs_train)
    shuffle(graphs_train)
    validation_data = graphs_train[0:int(params.ratio_validate*num_train_samples)]
    training_data = graphs_train[int(params.ratio_validate*num_train_samples):]
    return training_data, validation_data



# Create pytorch DataLoader for the model
def create_dataloader(params, graphs_train, ordering):
    # split
    print('Dataloaders..')
    train_data, validate_data = train_validate_split(params, graphs_train)
    
    # train dataloader
    train_dataset = Graph_sequence_sampler(train_data, params, ordering)
    sample_prob_train = [1.0 / len(train_dataset) for i in range(len(train_dataset))]
    train_sample_strategy = sampler.WeightedRandomSampler(sample_prob_train,
                                                          num_samples=params.sample_batches*params.batch_size,
                                                          replacement=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params.batch_size, 
                                  num_workers=params.num_workers,
                                  sampler=train_sample_strategy)
    
    # validation dataloader
    validate_dataset = Graph_sequence_sampler(validate_data, params, ordering)
    sample_prob_validate = [1.0 / len(validate_dataset) for i in range(len(validate_dataset))]
    validate_sample_strategy = sampler.WeightedRandomSampler(sample_prob_validate,
                                                             num_samples=params.sample_batches*params.batch_size,
                                                             replacement=True)
    validate_dataloader = DataLoader(validate_dataset,
                                     batch_size=params.batch_size, 
                                     num_workers=params.num_workers,
                                     sampler=validate_sample_strategy)
    
    return train_dataloader, validate_dataloader



def size_dependent_sampling_prob(dataset):
    size_dict = {}
    for graph in dataset:
        size = int(graph['len'])
        if size not in size_dict:
            size_dict[size] = 1
        else:
            size_dict[size] += 1

    sampling_prob = list()
    for graph in dataset:
        size = int(graph['len'])
        sampling_prob.append( (np.sqrt(1/size_dict[size])) )    
        #sampling_prob.append( (1/size_dict[size]) )      
    sampling_prob /= np.sum(sampling_prob)

    return sampling_prob 


# Create pytorch DataLoader for the model
def create_dataloader_only_train(params, graphs_train, ordering, size_dependent_sampling=False):
    
    train_dataset = Graph_sequence_sampler(graphs_train, params, ordering)
    if size_dependent_sampling:
        sample_prob_train = size_dependent_sampling_prob(train_dataset)
    else:
        sample_prob_train = [1.0 / len(train_dataset) for i in range(len(train_dataset))]

    train_sample_strategy = sampler.WeightedRandomSampler(sample_prob_train,
                                                          num_samples=params.sample_batches*params.batch_size,
                                                          replacement=True)
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=params.batch_size,
                                  num_workers=params.num_workers,
                                  sampler=train_sample_strategy)
    
    return train_dataloader



###########################################################################################################################

def convert_X_to_set(X):
    nodes = dict()
    for obj in X:
        obj = int(obj)-1
        if obj not in nodes:
            nodes[obj] = int(1)
        else:
            nodes[obj] += int(1)
    nodes = {k: v for k, v in sorted(nodes.items(), key=lambda item: item[0], reverse=False)}
    return (list(nodes.keys()), list(nodes.values()))


class Node_set_sampler(Dataset):
    """
    Custom dataset class for node set generation model
    """
    def __init__(self, graphs_all, params, ind_to_classes):
        self.one_hot = params.one_hot
        self.ind_to_classes = ind_to_classes
        self.nodes_all = []
        self.count_mlp_input = params.count_mlp_input
        self.max_count = params.max_count
        self.max_cardinality = params.max_cardinality
        self.category_dim = params.category_dim
        for graph in graphs_all:
            X, _ = graph
            data = convert_X_to_set(X)
            max_count = 1
            if max(data[1])>max_count:
                max_count = max(data[1])
            if max_count<self.max_count and len(data[0])<self.max_cardinality:
                self.nodes_all.append(data)

    def __len__(self):
        return len(self.nodes_all)
    
    def __getitem__(self, idx):
        # copy data
        objects = np.array(self.nodes_all[idx][0].copy())
        counts = np.array(self.nodes_all[idx][1].copy())
        n = len(objects)
        # Encoder input
        encoder_in_object_cat = np.zeros(self.max_cardinality)
        encoder_in_object_cat[:n] = objects+1
        encoder_in_object_count = np.zeros(self.max_cardinality)
        encoder_in_object_count[:n] = counts
        encoder_in_card = n
        # encoder_in = np.zeros((self.category_dim, self.max_cardinality))
        # encoder_in[objects, np.arange(n)] = counts
        # encoder_in_object_cat = np.zeros((self.category_dim, self.max_cardinality))
        # encoder_in_object_cat[objects, np.arange(n)] = 1
        # encoder_in_object_count = np.zeros((self.max_count, self.max_cardinality))
        # encoder_in_object_count[np.array(counts)-1, np.arange(n)] = 1
        # encoder_in_card = np.zeros(self.max_cardinality)
        # encoder_in_card[n] = 1
        
        # decoder output
        cardinality = n-1
        decoder_out_object_cat = np.zeros(self.category_dim)
        decoder_out_object_cat[objects] = 1
        # decoder_out_object_count = -1*np.ones(self.category_dim)
        # decoder_out_object_count[np.array(objects)] = np.array(counts)-1
        decoder_out_object_count = -1*np.ones(self.max_cardinality)
        decoder_out_object_count[:n] = counts
        # decoder input (count mlp)
        decoder_in_object_count = np.zeros((self.count_mlp_input, self.max_cardinality))
        # multi-label encoding of label set
        decoder_in_object_count[objects, :n] = 1
        # one-hot encoding of current object
        decoder_in_object_count[self.category_dim+objects, np.arange(n)] = 1
        # multi-label encoding of counts (autoregressive)
        for i in range(n-1):
            object_count = np.zeros(self.category_dim)
            np.put(object_count, objects[:i], counts[:i])
            decoder_in_object_count[2*self.category_dim:3*self.category_dim, i+1] = object_count
        return {
            #'encoder_in': encoder_in,
            'encoder_in_object_cat': encoder_in_object_cat,
            'encoder_in_object_count': encoder_in_object_count,
            'encoder_in_cardinality': encoder_in_card,
            'decoder_out_cardinality': cardinality,
            'decoder_in_object_count': decoder_in_object_count,
            'decoder_out_object_cat': decoder_out_object_cat,
            'decoder_out_object_count': decoder_out_object_count
        }



def create_dataloader_VAE(params, graphs_train, ind_to_classes):
    # create dataset
    dataset = Node_set_sampler(graphs_train, params, ind_to_classes)
    # dataloaders
    dataloader = DataLoader(dataset,
                                  batch_size=params.batch_size,
                                  num_workers=params.num_workers,
                                  shuffle=True)
    # val_dataloader = DataLoader(dataset,
    #                             batch_size=params.batch_size,
    #                             num_workers=params.num_workers,
    #                             #xshuffle=True,
    #                             drop_last=True,
    #                             sampler=torch.utils.data.SubsetRandomSampler(indices[int(0.8*n):]))
    
    return dataloader
