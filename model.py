from imports import *
from FSpool import *
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class GRU_graph(nn.Module):
    """
    The rnn which computes graph-level hidden state at each step i, 
    from X(i-1), F(i-1) and hidden state from edge level rnn.
    1 layer at input, rnn with num_layers, output hidden state
    """
    def __init__(self, max_num_node,
                 input_size, embedding_size,
                 hidden_size, num_layers,
                 bias_constant):
        super(GRU_graph, self).__init__()

        # Define the architecture
        self.max_num_node = max_num_node
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        # input
        self.input = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )
        # rnn
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        # initialialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant)
            elif 'weight' in name:
                #nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    
    # Initial state of GRU_graph is 0.
    def init_hidden(self, batch_size, random_noise=False):
        hidden_init = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(DEVICE)
        if random_noise:
            std = 0.05
            noise_sampler = torch.distributions.MultivariateNormal(torch.zeros(self.hidden_size), std*torch.eye(self.hidden_size))
            hidden_init[0] = noise_sampler.sample((batch_size,))

        return hidden_init

    def forward(self, input_raw, input_len, pack=False):
        # input
        #print('input_raw', input_raw.shape)
        input_ggru = self.input(input_raw)
        #print('input_ggru', input_ggru.shape, input_ggru)
        if pack:
            input_packed = pack_padded_sequence(input_ggru, input_len, batch_first=True, enforce_sorted=False)
        else:
            input_packed = input_ggru
        # rnn
        output_raw, self.hidden = self.rnn(input_packed, self.hidden)
        if pack:
            output_raw, seq_len = pad_packed_sequence(output_raw, batch_first=True, padding_value=0.0, 
                                                      total_length=self.max_num_node)
        #print('output_raw', output_raw.shape, output_raw)
        return output_raw
    


class MLP_node(nn.Module):
    """
    2 layer Multilayer perceptron with sigmoid output to get node categories.
    2 layered fully connected with ReLU
    """
    def __init__(self, h_graph_size, embedding_size, node_size):
        super(MLP_node, self).__init__()

        self.output = nn.Sequential(
            nn.Linear(h_graph_size, embedding_size),
            nn.ReLU(),
            nn.Linear(embedding_size, node_size)
        )
        ## initialialization
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('sigmoid'))
                #m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')

    def forward(self, h):
        y = self.output(h)
        return y

    
# class GRU_edge(nn.Module):
#     """
#     Sequential NN which outputs the edge categories F(i) using GRU_graph hidden state and X(i)
#     1 layer at input, rnn with hidden layers, 2 layer at output
#     """
#     def __init__(self,
#                  input_size, embedding_size,
#                  h_edge_size, num_layers,
#                  emb_edge_size, edge_size,
#                  bias_constant):
#         super(GRU_edge, self).__init__()
        
#         ## Define the architecture
#         self.num_layers = num_layers
#         self.hidden_size = h_edge_size
#         # input
#         self.input = nn.Sequential(
#             nn.Linear(input_size, embedding_size),
#             nn.ReLU()
#         )
#         # gru
#         self.rnn = nn.GRU(input_size=embedding_size, hidden_size=h_edge_size, 
#                           num_layers=num_layers, batch_first=True)
#         # outputs from the gru
#         self.output_to = nn.Sequential(
#                 nn.Linear(h_edge_size, emb_edge_size),
#                 nn.ReLU(),
#                 nn.Linear(emb_edge_size, edge_size)
#             )
#         self.output_from = nn.Sequential(
#                 nn.Linear(h_edge_size, emb_edge_size),
#                 nn.ReLU(),
#                 nn.Linear(emb_edge_size, edge_size)
#             )
#         # initialialization
#         self.hidden = None
#         for name, param in self.rnn.named_parameters():
#             if 'bias' in name:
#                 nn.init.constant_(param, bias_constant)
#             elif 'weight' in name:
#                 nn.init.xavier_uniform_(param, gain=nn.init.calculate_gain('sigmoid'))
#                 #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
#                 #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
    
#     def forward(self, input_raw):
#         # input
#         input_egru = self.input(input_raw)
#         # rnn
#         output_raw, self.hidden = self.rnn(input_egru, self.hidden)
#         # output
#         output_from = self.output_from(output_raw)
#         output_to = self.output_to(output_raw)
        
#         return output_from, output_to

    
class GRU_edge_ver2(nn.Module):
    """
    Sequential NN which outputs the edge categories F(i) using GRU_graph hidden state and X(i)
    1 layer at input, rnn with hidden layers, 2 layer at output
    """
    def __init__(self,
                 input_size, embedding_size,
                 h_edge_size, num_layers,
                 emb_edge_size, edge_size,
                 bias_constant):
        super(GRU_edge_ver2, self).__init__()
        
        ## Define the architecture
        self.num_layers = num_layers
        self.hidden_size = h_edge_size
        # input
        self.input = nn.Sequential(
            nn.Linear(input_size, embedding_size),
            nn.ReLU()
        )
        # gru
        self.rnn = nn.GRU(input_size=embedding_size, hidden_size=h_edge_size, 
                          num_layers=num_layers, batch_first=True)
        # outputs from the gru
        self.output= nn.Sequential(
                nn.Linear(h_edge_size, emb_edge_size),
                nn.ReLU(),
                nn.Linear(emb_edge_size, edge_size)
            )
        # initialialization
        self.hidden = None
        for name, param in self.rnn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, bias_constant)
            elif 'weight' in name:
                #nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('sigmoid'))
                nn.init.xavier_uniform_(param,gain=nn.init.calculate_gain('relu'))
                #nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data = init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain('relu'))
                #m.weight.data = init.kaiming_uniform_(m.weight.data, mode='fan_in', nonlinearity='relu')
                
    
    def forward(self, input_raw):
        # input
        input_egru = self.input(input_raw)
        # rnn
        output_raw, self.hidden = self.rnn(input_egru, self.hidden)
        # output
        output = self.output(output_raw)
        
        return output


############################ Transformer Model #####################################

def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GraphTransformerEncoder(nn.Module):

    def __init__(self, in_size, nhead, hidden_size, nlayers, out_size, dropout=0.1):
        super(GraphTransformerEncoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_size, dropout)
        encoder_layers = TransformerEncoderLayer(in_size, nhead, hidden_size, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.linear = nn.Linear(in_size, out_size)
    def forward(self, src, src_mask=None):
        src = self.pos_encoder(src)
        if src_mask is not None:
            graph_enc = self.transformer_encoder(src, src_mask)
        else:
            graph_enc = self.transformer_encoder(src)
        graph_mem = self.linear(graph_enc)
        return graph_mem


class EdgeTransformerDecoder(nn.Module):

    def __init__(self, in_size, nhead, hidden_size, nlayers, emb_edge_size, edge_size, dropout=0.1):
        super(EdgeTransformerDecoder, self).__init__()
        
        self.pos_encoder = PositionalEncoding(in_size, dropout)
        decoder_layers = TransformerDecoderLayer(in_size, nhead, hidden_size, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, nlayers)
        self.linear = nn.Sequential(
                      nn.Linear(in_size, emb_edge_size),
                      nn.ReLU(),
                      nn.Linear(emb_edge_size, edge_size)
                      )

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        tgt = self.pos_encoder(tgt)
        if tgt_mask is not None and memory_mask is not None:
            output = self.transformer_decoder(tgt, memory, tgt_mask, memory_mask)
        else:
            output = self.transformer_decoder(tgt, memory)
        edge_logit = self.linear(output)
        return edge_logit

 