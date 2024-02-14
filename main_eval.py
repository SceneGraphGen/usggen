from train import *
from generate import *
from evaluate import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--num_graphs', default=2000, type=int)
parser.add_argument('--num_eval', default=1000, type=int)
parser.add_argument('--gen_data', default=1, type=int)
parser.add_argument('--hyperparam_str', default='usggen_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710', type=str)
args = parser.parse_args()

if __name__ == "__main__":
    print('Loading data..')
    RANDOM_SEED = 121
    os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)
    print("random seed", RANDOM_SEED)

    graphs_train = pickle.load(open(os.path.join('./data','train_dataset.p'), 'rb'))
    graphs_test = pickle.load(open(os.path.join('./data','test_dataset.p'), 'rb'))
    ind_to_classes, ind_to_predicates, _ = pickle.load(open(os.path.join('./data','categories.p'), 'rb'))
    print('Number of training samples: ', len(graphs_train))
    print('Number of test samples: ', len(graphs_test))

    hyperparam_str = args.hyperparam_str

    print('Loading SceneGraphGen model')
    # generate graphs
    hyperparam_str = args.hyperparam_str # 'sggen_ordering-random_classweights-none_nodepred-True_edgepred-True_argmaxFalse_MHPFalse_batch-256_samples-256_epochs-300_nlr-0.001_nlrdec-0.95_nstep-1710_elr-0.001_elrdec-0.95_estep-1710'
    params, config, models = load_models('./models', hyperparam_str)
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    gen_path = os.path.join('./generated_samples/', hyperparam_str)
    if args.gen_data:
        print('Generating graph dataset..')
        class_dict, _ = prior_distribution(graphs_train, ordering)
        graphs_data = generate_scene_graphs(gen_path, params, config, models, args.num_graphs, class_dict, ind_to_classes, ind_to_predicates, make_visuals=True)
        pickle.dump(graphs_data, open(os.path.join(gen_path, 'generated_objects.p'), 'wb') )
    else:
        print('Loading graph dataset..')
        graphs_data = pickle.load(open(os.path.join(gen_path, 'generated_objects.p'), 'rb'))
    node_data = get_node_set(graphs_data)

    os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    print('MMD evaluation of nodes..')
    test_node_data = get_node_set(graphs_test)
    shuffle(test_node_data)
    shuffle(node_data)    
    t1 = time.time()
    test_gen_mmd = compute_mmd(node_data[:args.num_eval], test_node_data[:args.num_eval], dirac_set_kernel)
    t2 = time.time()
    print('MMD b/w test-generated on node set kernel: ', test_gen_mmd, '. Time taken: ', t2-t1)

    print('MMD evaluation of graphs..')
    shuffle(graphs_test)
    shuffle(graphs_data)
    t1 = time.time()
    test_gen_mmd = compute_mmd(graphs_data[:args.num_eval], graphs_test[:args.num_eval], dirac_random_walk_kernel)
    t2 = time.time()
    print('MMD b/w test-generated on graph kernel: ', test_gen_mmd, '. Time taken: ', t2-t1)
    