from utils import *

# Define the parameters of the model
class Model_params():
    def __init__(self, batch_size, sample_batches, 
                node_lr_init, node_lr_end, node_lr_decay, 
                edge_lr_init, edge_lr_end, edge_lr_decay,
                num_epochs, reg, bias_constant, config,
                small_network, edge_supervision=True):
        
        print('Setting model parameters..')
        # num threads for DataLoader
        self.num_workers = 1
        self.batch_size = batch_size
        self.sample_batches = sample_batches
        self.epochs = num_epochs
        self.reg = reg
        self.bias_constant = bias_constant
        # number of graphs in eval
        self.num_graphs_eval = 100
        # number of generators for MHP
        self.num_generators = 5
        ####### NODE GENERATOR ##########
        # learning rate and epochs
        self.node_lr_init = node_lr_init
        self.node_lr_end = node_lr_end
        self.node_lr_decay = node_lr_decay
        self.node_lr = node_lr_init
        self.node_lr_rate = node_lr_decay
        node_decay_steps = log(node_lr_decay)*num_epochs*sample_batches/log((node_lr_end/node_lr_init))
        if node_decay_steps < 1:
            print('Decay rate too fine')
            self.node_step_decay_epochs = 1
        else:
            self.node_step_decay_epochs = int(node_decay_steps)
        ####### EDGE GENERATOR ##########
        # learning rate and epochs
        self.edge_lr_init = edge_lr_init
        self.edge_lr_end = edge_lr_end
        self.edge_lr_decay = edge_lr_decay
        self.edge_lr = edge_lr_init
        self.edge_lr_rate = edge_lr_decay
        edge_decay_steps = log(edge_lr_decay)*num_epochs*sample_batches/log((edge_lr_end/edge_lr_init))
        if edge_decay_steps < 1:
            print('Decay rate too fine')
            self.edge_step_decay_epochs = 1
        else:
            self.edge_step_decay_epochs = int(edge_decay_steps)
        
        # VISUAL GENOME DATA
        # lengths of rnn models
        self.min_num_node = 3
        self.max_num_node = 25
        self.max_edge_len = 25
        # node/edge categories
        self.num_node_categories = 151
        self.num_edge_categories = 53
        self.node_EOS_token = 150
        self.edge_SOS_token = 52
        self.no_edge_token = 51
        # SGNET DATA
        # # lengths of rnn models
        # self.min_num_node = 10
        # self.max_num_node = 40
        # self.max_edge_len = 40
        # # node/edge categories
        # self.num_node_categories = 51
        # self.num_edge_categories = 8
        # self.node_EOS_token = 50
        # self.edge_SOS_token = 7
        # self.no_edge_token = 6
        
        edge_seq_len = self.max_num_node
        self.small_network = small_network
        # ------------network parameters for small dataset --------------------------------------------
        if small_network:
            # embedding
            self.node_emb_size = 64
            self.edge_emb_size = 8
            # GRU_graph
            self.ggru_input_size = 2*self.edge_emb_size*(edge_seq_len-1) + self.node_emb_size
            self.ggru_emb_size = 128
            self.ggru_hidden_size = 64
            self.ggru_num_layers = 2
            # MLP_node
            self.mlp_input_size = self.ggru_hidden_size
            self.mlp_emb_size = 256
            self.mlp_out_size = self.num_node_categories
            # GRU_edge
            if edge_supervision:
                self.egru_input_size = 2*self.edge_emb_size + 2*self.node_emb_size
            else:
                self.egru_input_size = 2*self.edge_emb_size
            self.egru_hidden_size = self.ggru_hidden_size
            self.egru_emb_input_size = 64
            self.egru_num_layers = 2
            self.egru_emb_output_size = 64
            self.egru_output_size = self.num_edge_categories-2
        # ----- network parameters for big dataset ---------------------------------------------------
        else:
            # embedding
            self.node_emb_size = 64
            self.edge_emb_size = 8
            # GRU_graph
            self.ggru_input_size = 2*self.edge_emb_size*(self.max_edge_len-1) + self.node_emb_size
            # self.ggru_input_size = 2*self.edge_emb_size*(edge_seq_len-1) + self.node_emb_size
            self.ggru_emb_size = 512
            self.ggru_hidden_size = 128
            self.ggru_num_layers = 4
            # MLP_node
            self.mlp_input_size = self.ggru_hidden_size
            self.mlp_emb_size = 256
            self.mlp_out_size = self.num_node_categories
            # GRU_edge
            if edge_supervision:
                if EDGE_INFO:
                    self.egru_input_size1 = 3*self.edge_emb_size + 2*self.node_emb_size
                else:
                    self.egru_input_size1 = 2*self.edge_emb_size + 2*self.node_emb_size
                self.egru_input_size2 = 2*self.edge_emb_size + 2*self.node_emb_size
            else:

                self.egru_input_size1 = 2*self.edge_emb_size
                self.egru_input_size2 = 2*self.edge_emb_size
            self.egru_hidden_size = self.ggru_hidden_size
            self.egru_emb_input_size = 256
            self.egru_num_layers = 4
            self.egru_emb_output_size = 128
            self.egru_output_size = self.num_edge_categories-2




class Model_params_VAE():
    def __init__(self, batch_size, lr, num_epochs):
        print('Setting model parameters..')
        self.batch_size = batch_size
        self.epochs = num_epochs
        self.lr = lr
        self.warmup_epochs = 30
        self.num_workers = 1
        self.one_hot = True
        self.max_cardinality = 18
        self.max_count = 8
        if self.one_hot:
            self.category_dim = 150
        else:
            self.category_dim = 100
        # Encoder network parameters
        self.cardinality_emb_size = 8
        self.category_emb_size = 32
        self.count_emb_size = 8
        #self.phi_in = self.category_dim + self.max_count
        self.cat_phi_in = self.category_emb_size
        self.cat_phi_hidden = 64
        self.cat_phi_out = 128
        self.count_phi_in = self.count_emb_size 
        self.count_phi_hidden = 64
        self.count_phi_out = 128
        self.rho_hidden = 256
        self.rho_out = 128
        self.num_latent_samples = 1
        # Decoder network parameters
        self.latent_size = 32
        self.cat_mlp_hidden = 128
        self.cat_mlp_out = 256
        self.card_mlp_hidden = 64
        self.card_mlp_out = 128
        self.count_mlp_input = 3*self.category_dim
        self.count_mlp_emb = 128
        # self.count_mlp_hidden_size = 512
        self.count_mlp_hidden = 256
        # number of samples per batch
        self.gen_batch = 20
        self.regularization = 0.0
        self.beta = 0.2
        self.early_stop = False
        self.patience = 15
        self.epochs_viz_z = 5