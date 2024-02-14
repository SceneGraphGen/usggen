from train import *
from generate import *
from evaluate import *

from torch.utils.tensorboard import SummaryWriter
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--run_folder', default='new_usggen_', type=str)
args = parser.parse_args()

def tune_model(graphs_train, graphs_test, hyperparams, run_str, config, class_weights, ind_to_classes,
               ind_to_predicates, small_network):
    # train for each hyperparameter setting
    print('Begin model tuning')
    model_dict = {}
    for hyperparam in hyperparams:
        ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
        batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay, edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant = hyperparam
        _, _, _, _ = class_weights
        # set parameters
        # print('Setting model parameters..')
        params = Model_params(batch_size, sample_batches, node_lr_init, node_lr_end, node_lr_decay,
                            edge_lr_init, edge_lr_end, edge_lr_decay, num_epochs, reg, bias_constant,
                            config, small_network)
        hyperparam_str = 'ordering-'+str(ordering)+'_classweights-'+str(weighted_loss)+'_nodepred-'+str(node_pred)+'_edgepred-'+str(edge_pred)\
                        +'_argmax'+str(use_argmax) + '_MHP'+str(use_MHP)+'_batch-'+str(batch_size) + '_samples-'+str(sample_batches) + '_epochs-'+str(num_epochs)\
                        +'_nlr-'+str(params.node_lr) + '_nlrdec-'+ str(params.node_lr_rate) + '_nstep-'+str(params.node_step_decay_epochs)\
                        +'_elr-'+str(params.edge_lr) + '_elrdec-'+ str(params.edge_lr_rate) + '_estep-'+str(params.edge_step_decay_epochs)
        writer = SummaryWriter('./runs/'+run_str+hyperparam_str)
        # instantiate models
        models = instantiate_model_classes(params, config)
        # create dataloader
        graph_dataloader = create_dataloader_only_train(params, graphs_train, ordering)
        # train model
        trained_model = train(params, config, writer, graph_dataloader, models, run_str+hyperparam_str, class_weights)
        # generate samples
        path = os.path.join('./generated_samples/', run_str+hyperparam_str)
        os.makedirs(path, exist_ok=True)
        shuffle(graphs_train)
        seed_graphs = graphs_train[0:30]
        class_dict, _ = prior_distribution(graphs_train, ordering)
        #generate_nodes_sg(path, params, config, seed_graphs, models, 20, ind_to_classes, ind_to_predicates)
        if node_pred and not edge_pred:
            generate_scene_graphs_given_edges(path, params, config, seed_graphs, models, 20, ind_to_classes, ind_to_predicates)
        elif edge_pred and not node_pred:
            generate_scene_graphs_given_nodes(path, params, config, seed_graphs, trained_model, 20, ind_to_classes, ind_to_predicates)
        else:
            num_graphs = params.num_graphs_eval
            graphs_gen = generate_scene_graphs(path, params, config, models, num_graphs, class_dict, ind_to_classes, ind_to_predicates)
            sg = Digraph('sg', format='png')
            Xin, Fin = graphs_train[0]
            Fto = np.triu(Fin, +1)
            Ffrom = np.transpose(np.tril(Fin, -1))
            for idx, node in enumerate(Xin-1):
            #for idx, node in enumerate(Xin):
                sg.node(str(idx), ind_to_classes[node])
            to_edge_obj, to_edge_subj = np.nonzero(Fto)
            for obj, subj in zip(to_edge_obj, to_edge_subj):
                sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj]-1)])
                #sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Fto[obj, subj])])
            from_edge_subj, from_edge_obj = np.nonzero(Ffrom)
            for obj, subj in zip(from_edge_obj, from_edge_subj):
                sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj]-1)])
                #sg.edge(str(obj), str(subj), label=ind_to_predicates[int(Ffrom[subj, obj])])
            sg.render(os.path.join(path, 'Ground_truth'), view=False)

    print('Done.')



if __name__ == "__main__":
    print('Loading data..')

    RANDOM_SEED = 42
    os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    graphs_train = pickle.load(open(os.path.join('./data','train_dataset.p'), 'rb'))
    graphs_test = pickle.load(open(os.path.join('./data','test_dataset.p'), 'rb'))
    ind_to_classes, ind_to_predicates, _ = pickle.load(open(os.path.join('./data','categories.p'), 'rb'))
    print('Number of training samples: ', len(graphs_train))
    print('Number of test samples: ', len(graphs_test))
    params = Model_params(batch_size=1, sample_batches=1, node_lr_init=0.001, node_lr_end=0.0001, node_lr_decay=0.95, 
                            edge_lr_init=0.001, edge_lr_end=0.0001, edge_lr_decay=0.95, num_epochs=1000, reg=0, bias_constant=0.25,
                            config=None, small_network=True)
    node_freq = get_node_freq(params, graphs_train)
    edge_freq = get_edge_freq(params, graphs_train)
    INV_NODE_WEIGHT, INV_EDGE_WEIGHT = inverse_freq_weights(params, node_freq, edge_freq)
    SQRT_INV_NODE_WEIGHT, SQRT_INV_EDGE_WEIGHT = sqrt_inverse_freq_weights(params, node_freq, edge_freq)
    class_weights = INV_NODE_WEIGHT, INV_EDGE_WEIGHT, SQRT_INV_NODE_WEIGHT, SQRT_INV_EDGE_WEIGHT
    class_weights = [0, 0, 0, 0]

    # set hyperparameters
    hyperparams_list = [[] for _ in range(11)]
    hyperparams_list[0] = [256] # batch size
    hyperparams_list[1] = [256] # sample batches per epoch
    hyperparams_list[2] = [0.001] # node initial lerning rate
    hyperparams_list[3] = [0.0001] # node final lerning rate
    hyperparams_list[4] = [0.95] # node lr decay
    hyperparams_list[5] = [0.001] # edge initial lerning rate
    hyperparams_list[6] = [0.0001] # edge final lerning rate
    hyperparams_list[7] = [0.95] # edge lr decay
    hyperparams_list[8] = [300] # number of epochs
    # fixed parameters
    hyperparams_list[9] = [0.0] # regularization
    hyperparams_list[10] = [0.25] # bias initialization
    hyperparams = list(itertools.product(*hyperparams_list))
    # set model configuration
    config = {}
    config['order'] = 'random'
    config['class_weight'] = 'none'
    config['node_pred'] = True
    config['edge_pred'] = True
    config['use_argmax'] = False
    config['use_MHP'] = False
    small_network = False
    # atleast one must be true
    assert config['node_pred'] | config['edge_pred'] == True
    
    # tune model
    shuffle(graphs_train)
    tune_model(graphs_train, graphs_test, hyperparams, args.run_folder, config, class_weights, 
               ind_to_classes, ind_to_predicates, small_network)
