from model_params import *
from model import *


def get_config(config):
    ordering = config['order']
    weighted_loss = config['class_weight']
    node_pred = config['node_pred']
    edge_pred = config['edge_pred']
    use_argmax = config['use_argmax']
    use_MHP = config['use_MHP']
    return ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP


def set_config(config_params):
    config = {}
    config['order'] = config_params[0]
    config['class_weight'] = config_params[1]
    config['node_pred'] = config_params[2]
    config['edge_pred'] = config_params[3]
    config['use_argmax'] = config_params[4]
    config['use_MHP'] = config_params[5]
    return config


def plot_loss(train_loss, name):
    epochs = np.arange(len(train_loss))
    plt.plot(epochs, train_loss, 'b', label='training loss')
    plt.legend(loc="upper right")
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.savefig('./figures/'+ name+ '.png')
    plt.clf()
    plt.cla()
    plt.close()


def instantiate_model_classes(params, config, use_glove=False):
    print('Initialize the model..')
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    # Embeddings for input
    node_emb = nn.Embedding(params.num_node_categories,
                            params.node_emb_size, 
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE) # 0, 1to150
    if use_glove:
        glove_emb = torch.Tensor(pickle.load(open(os.path.join('./data','glove_emb.p'), 'rb')))
        node_emb.load_state_dict({'weight': glove_emb})
    node_emb.weight.requires_grad=False
    edge_emb = nn.Embedding(params.num_edge_categories,
                            params.edge_emb_size,
                            padding_idx=0,
                            scale_grad_by_freq=False).to(DEVICE) # 0, 1to50, 51, 52
    edge_emb.weight.requires_grad=False
    # Node Generator
    if node_pred:
        mlp_node = MLP_node(h_graph_size=params.mlp_input_size,
                                embedding_size=params.mlp_emb_size,
                                node_size=params.mlp_out_size).to(DEVICE)
        gru_graph3 = GRU_graph(max_num_node=params.max_num_node,
                           input_size=params.node_emb_size,
                           embedding_size=params.ggru_emb_size,
                           hidden_size=params.ggru_hidden_size,
                           num_layers=params.ggru_num_layers,
                           bias_constant=params.bias_constant).to(DEVICE)
    else:
        mlp_node = None
        gru_graph3 = None
    # Edge Generator
    if edge_pred:
        gru_graph1 =  GRU_graph(max_num_node=params.max_num_node,
                                input_size=params.ggru_input_size,
                                embedding_size=params.ggru_emb_size,
                                hidden_size=params.ggru_hidden_size,
                                num_layers=params.ggru_num_layers,
                                bias_constant=params.bias_constant).to(DEVICE)
        gru_graph2 =  GRU_graph(max_num_node=params.max_num_node,
                                input_size=params.ggru_input_size,
                                embedding_size=params.ggru_emb_size,
                                hidden_size=params.ggru_hidden_size,
                                num_layers=params.ggru_num_layers,
                                bias_constant=params.bias_constant).to(DEVICE)

        gru_edge1 = GRU_edge_ver2(input_size=params.egru_input_size1,
                                embedding_size=params.egru_emb_input_size,
                                h_edge_size=params.egru_hidden_size,
                                num_layers=params.egru_num_layers,
                                emb_edge_size=params.egru_emb_output_size,
                                edge_size=params.egru_output_size,
                                bias_constant=params.bias_constant).to(DEVICE)
        gru_edge2 = GRU_edge_ver2(input_size=params.egru_input_size2,
                                embedding_size=params.egru_emb_input_size,
                                h_edge_size=params.egru_hidden_size,
                                num_layers=params.egru_num_layers,
                                emb_edge_size=params.egru_emb_output_size,
                                edge_size=params.egru_output_size,
                                bias_constant=params.bias_constant).to(DEVICE)
    else:
        gru_graph1 = None
        gru_graph2 = None
        gru_edge1 = None
        gru_edge2 = None

    return node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2


def save_models(models, path, hyperparam_str, config, params):
    print('Saving model..')
    folder = os.path.join(path, hyperparam_str)
    os.makedirs(folder, exist_ok=True)
    # save config
    config_params = get_config(config)
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = config_params
    pickle.dump(config_params, open(os.path.join(folder,"config.p"), "wb"))
    # save params
    model_params = params.batch_size, params.sample_batches, params.node_lr_init, params.node_lr_end, params.node_lr_decay,\
                   params.edge_lr_init, params.edge_lr_end, params.edge_lr_decay, params.epochs, params.reg, params.bias_constant, params.small_network
    pickle.dump(model_params, open(os.path.join(folder,"params.p"), "wb"))
    # save models
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    fname = os.path.join(folder, 'node_emb.dat')
    torch.save(node_emb.state_dict(), fname)
    fname = os.path.join(folder, 'edge_emb.dat')
    torch.save(edge_emb.state_dict(), fname)
    if node_pred:
        fname = os.path.join(folder, 'MLP_node.dat')
        torch.save(mlp_node.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_graph3.dat')
        torch.save(gru_graph3.state_dict(), fname)
    if edge_pred:
        fname = os.path.join(folder, 'GRU_graph1.dat')
        torch.save(gru_graph1.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_graph2.dat')
        torch.save(gru_graph2.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_edge1.dat')
        torch.save(gru_edge1.state_dict(), fname)
        fname = os.path.join(folder, 'GRU_edge2.dat')
        torch.save(gru_edge2.state_dict(), fname)
    

def load_models(path, hyperparam_str):
    print('Loading trained model..')
    folder = os.path.join(path, hyperparam_str)
    # load config
    config_params = pickle.load(open(os.path.join(folder,"config.p"), 'rb'))
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = config_params
    config = set_config(config_params)
    # load params
    model_params = pickle.load(open(os.path.join(folder,"params.p"), 'rb'))
    batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay, edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant, small_network = model_params
    params = Model_params(batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay,
                            edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant,
                            config, small_network)
    # load models
    models = instantiate_model_classes(params, config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    fname = os.path.join(folder, 'node_emb.dat')
    node_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    fname = os.path.join(folder, 'edge_emb.dat')
    edge_emb.load_state_dict(torch.load(fname, map_location=DEVICE))
    if node_pred:
        fname = os.path.join(folder, 'MLP_node.dat')
        mlp_node.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_graph3.dat')
        gru_graph3.load_state_dict(torch.load(fname, map_location=DEVICE))
    if edge_pred:
        fname = os.path.join(folder, 'GRU_graph1.dat')
        gru_graph1.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_graph2.dat')
        gru_graph2.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_edge1.dat')
        gru_edge1.load_state_dict(torch.load(fname, map_location=DEVICE))
        fname = os.path.join(folder, 'GRU_edge2.dat')
        gru_edge2.load_state_dict(torch.load(fname, map_location=DEVICE))

    models = node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2
    return params, config, models
