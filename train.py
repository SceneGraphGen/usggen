from train_helper import *
from data import *

def focal_loss(CEloss, alpha, gamma):
    pt = torch.exp(-CEloss)
    focal_loss = (alpha * (1-pt)**gamma * CEloss)
    return focal_loss


def forward_pass_model(params, config, data, models, node_CELoss, edge_CELoss,
                       edge_supervision=True, use_focal_loss=False, size_dependent_edge_loss=False):
    
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # focal_loss parameters
    alpha = 1
    gamma = 2
    # INPUTS
    # GRU_graph
    Xin_ggru = data['Xin_ggru'].to(DEVICE).long()
    Fto_in_ggru = data['Fto_in_ggru'].to(DEVICE).long()
    Ffrom_in_ggru = data['Ffrom_in_ggru'].to(DEVICE).long()
    # GRU_edge
    Xin_egru1 = data['Xin_egru1'].to(DEVICE).long()
    Xin_egru2 = data['Xin_egru2'].to(DEVICE).long()
    Fto_in_egru = data['Fto_in_egru'].to(DEVICE).long()
    Ffrom_in_egru = data['Ffrom_in_egru'].to(DEVICE).long()
    Fto_in_egru_shifted = data['Fto_in_egru_shifted'].to(DEVICE).long()
    # OUTPUTS
    # MLP_node
    Xout_mlp = data['Xout_mlp'].to(DEVICE).long()
    # GRU_edge
    Fto_out_egru = data['Fto_out_egru'].to(DEVICE).long()
    Ffrom_out_egru = data['Ffrom_out_egru'].to(DEVICE).long()
    seq_len = data['len'].to(DEVICE).float()
    num_edges = data['num_edges'].to(DEVICE).float()

    # -------------------RUN GRU_graph-----------------------
    # input = concatenated X, F_to, F_from
    Xin_ggru = node_emb(Xin_ggru)
    Fto_in_ggru = edge_emb(Fto_in_ggru) 
    Fto_in_ggru = Fto_in_ggru.contiguous().view(Fto_in_ggru.shape[0],
                                                Fto_in_ggru.shape[1], -1)
    Ffrom_in_ggru = edge_emb(Ffrom_in_ggru) 
    Ffrom_in_ggru = Ffrom_in_ggru.contiguous().view(Ffrom_in_ggru.shape[0],
                                                    Ffrom_in_ggru.shape[1], -1)
    gru_graph_input = torch.cat((Xin_ggru, Fto_in_ggru, Ffrom_in_ggru), 2)
    # initial hidden state gru_graph
    # run the GRU_graph
    if node_pred:
        gru_graph3.hidden = gru_graph3.init_hidden(batch_size=params.batch_size)
        hg3 = gru_graph3(Xin_ggru, input_len=seq_len)
    if edge_pred:
        gru_graph1.hidden = gru_graph1.init_hidden(batch_size=params.batch_size)
        gru_graph2.hidden = gru_graph2.init_hidden(batch_size=params.batch_size)
        hg1 = gru_graph1(gru_graph_input, input_len=seq_len)
        hg2 = gru_graph2(gru_graph_input, input_len=seq_len)
    # ----------------RUN MLP_node---------------------------
    if node_pred:
        X_pred = mlp_node(hg3)
        X_pred = X_pred.permute(0, 2, 1)
        node_loss = node_CELoss(X_pred, Xout_mlp)
        if use_focal_loss:
            node_loss = focal_loss(node_loss, alpha, gamma)
        node_loss = torch.sum(node_loss)/torch.sum(seq_len)
    else:
        node_loss = torch.Tensor([0.0])
    # ---------------RUN GRU_edge----------------------------
    # Last node produces EOS. for last step, GRU_edge is not run
    if edge_pred:
        edge_seq_len = seq_len-1
        Xin_egru1 = node_emb(Xin_egru1)
        Xin_egru2 = node_emb(Xin_egru2)
        Fto_in_egru = edge_emb(Fto_in_egru)
        Ffrom_in_egru = edge_emb(Ffrom_in_egru)
        Fto_in_egru_shifted = edge_emb(Fto_in_egru_shifted)
        if edge_supervision:
            gru_edge_input1 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru, Fto_in_egru_shifted), 3)
            gru_edge_input2 = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru), 3)
        else:
            gru_edge_input = torch.cat((Fto_in_egru, Ffrom_in_egru), 3)
        # merge 2nd dimension into batch dimension by packing 
        gru_edge_input1 = pack_padded_sequence(gru_edge_input1, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data
        gru_edge_input2 = pack_padded_sequence(gru_edge_input2, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data
        edge_batch_size = gru_edge_input1.shape[0]
        # initial hidden state for gru_edge
        gru_edge_hidden1 = hg1[:, 0:params.max_num_node-1, :]
        gru_edge_hidden2 = hg2[:, 0:params.max_num_node-1, :]
        # merge 2nd dimension into batch dimension by packing
        gru_edge_hidden1 = pack_padded_sequence(gru_edge_hidden1, edge_seq_len, batch_first=True, 
                                                enforce_sorted=False).data
        gru_edge_hidden2 = pack_padded_sequence(gru_edge_hidden2, edge_seq_len, batch_first=True, 
                                                enforce_sorted=False).data
        gru_edge_hidden1 = torch.unsqueeze(gru_edge_hidden1, 0)
        gru_edge_hidden2 = torch.unsqueeze(gru_edge_hidden2, 0)
        if params.egru_num_layers>1:
            gru_edge_hidden1 = torch.cat((gru_edge_hidden1, torch.zeros(params.egru_num_layers-1, edge_batch_size,
                                                                        gru_edge_hidden1.shape[2]).to(DEVICE)), 0)
            gru_edge_hidden2 = torch.cat((gru_edge_hidden2, torch.zeros(params.egru_num_layers-1, edge_batch_size,
                                                                        gru_edge_hidden2.shape[2]).to(DEVICE)), 0)
        gru_edge1.hidden = gru_edge_hidden1
        gru_edge2.hidden = gru_edge_hidden2
        # run gru_edge
        Ffrom_pred = gru_edge1(gru_edge_input1)
        Fto_pred = gru_edge2(gru_edge_input2)
        Fto_out_egru = pack_padded_sequence(Fto_out_egru, edge_seq_len, batch_first=True,
                                            enforce_sorted=False).data
        Ffrom_out_egru = pack_padded_sequence(Ffrom_out_egru, edge_seq_len, batch_first=True,
                                              enforce_sorted=False).data
        # compute loss
        Fto_pred = Fto_pred.permute(0,2,1)
        Ffrom_pred = Ffrom_pred.permute(0,2,1)
        Fto_edge_loss = edge_CELoss(Fto_pred, Fto_out_egru)
        Ffrom_edge_loss = edge_CELoss(Ffrom_pred, Ffrom_out_egru)
        if use_focal_loss:
            Fto_edge_loss = focal_loss(Fto_edge_loss, alpha, gamma)
            Ffrom_edge_loss = focal_loss(Ffrom_edge_loss, alpha, gamma)
        if size_dependent_edge_loss:
            Ffrom_out_FG_mask = data['Ffrom_out_FG_mask'].to(DEVICE).float()
            Fto_out_FG_mask = data['Fto_out_FG_mask'].to(DEVICE).float()
            M_FG = torch.sum(data['num_FG_edges'].to(DEVICE).float())
            M_BG = torch.sum(num_edges) - M_FG
            Ffrom_out_FG_mask = pack_padded_sequence(Ffrom_out_FG_mask, edge_seq_len, batch_first=True, enforce_sorted=False).data
            Fto_out_FG_mask = pack_padded_sequence(Fto_out_FG_mask, edge_seq_len, batch_first=True, enforce_sorted=False).data
            Fto_edge_loss_FG = Fto_edge_loss*Fto_out_FG_mask
            Ffrom_edge_loss_FG = Ffrom_edge_loss*Ffrom_out_FG_mask
            edge_loss_FG = torch.sum(Fto_edge_loss_FG + Ffrom_edge_loss_FG)
            edge_loss_BG = torch.sum(Fto_edge_loss + Ffrom_edge_loss) - edge_loss_FG
            edge_loss_FG /= M_FG
            edge_loss_BG /= M_BG
            edge_loss = edge_loss_FG + (M_BG/M_FG)*edge_loss_BG
        else:
            edge_loss = Fto_edge_loss + Ffrom_edge_loss
            edge_loss = torch.sum(edge_loss)/torch.sum(num_edges)
    else:
        edge_loss = torch.Tensor([0.0])

    return node_loss, edge_loss



def forward_pass_model_transformer(params, config, data, models, node_CELoss, edge_CELoss, node_mask, edge_mask):
    
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, GTE1, GTE2, GTE3, gru_edge1, gru_edge2 = models
    # INPUTS
    # GRU_graph
    Xin_ggru = data['Xin_ggru'].to(DEVICE).long()
    Fto_in_ggru = data['Fto_in_ggru'].to(DEVICE).long()
    Ffrom_in_ggru = data['Ffrom_in_ggru'].to(DEVICE).long()
    # GRU_edge
    Xin_egru1 = data['Xin_egru1'].to(DEVICE).long()
    Xin_egru2 = data['Xin_egru2'].to(DEVICE).long()
    Fto_in_egru = data['Fto_in_egru'].to(DEVICE).long()
    Ffrom_in_egru = data['Ffrom_in_egru'].to(DEVICE).long()
    # OUTPUTS
    # MLP_node
    Xout_mlp = data['Xout_mlp'].to(DEVICE).long()
    # GRU_edge
    Fto_out_egru = data['Fto_out_egru'].to(DEVICE).long()
    Ffrom_out_egru = data['Ffrom_out_egru'].to(DEVICE).long()
    seq_len = data['len'].to(DEVICE).float()
    num_edges = data['num_edges'].to(DEVICE).float()

    # -------------------RUN GRU_graph-----------------------
    # input = concatenated X, F_to, F_from
    Xin_ggru = node_emb(Xin_ggru)
    Fto_in_ggru = edge_emb(Fto_in_ggru)
    Fto_in_ggru = Fto_in_ggru.contiguous().view(Fto_in_ggru.shape[0],
                                                Fto_in_ggru.shape[1], -1)
    Ffrom_in_ggru = edge_emb(Ffrom_in_ggru) 
    Ffrom_in_ggru = Ffrom_in_ggru.contiguous().view(Ffrom_in_ggru.shape[0],
                                                    Ffrom_in_ggru.shape[1], -1)
    GTE_input = torch.cat((Xin_ggru, Fto_in_ggru, Ffrom_in_ggru), 2).permute(1, 0, 2)
    # run the GRU_graph
    GTE_mask = node_mask
    GTE_mem1 = GTE1(GTE_input, GTE_mask)
    GTE_mem2 = GTE2(GTE_input, GTE_mask)
    GTE_mem3 = GTE3(GTE_input, GTE_mask)
    
    # ----------------RUN MLP_node---------------------------
    X_pred = mlp_node(GTE_mem3)
    X_pred = X_pred.permute(1, 2, 0)
    node_loss = node_CELoss(X_pred, Xout_mlp)
    node_loss = torch.sum(node_loss)/torch.sum(seq_len)

    # ---------------RUN GRU_edge----------------------------
    # Last node produces EOS. for last step, GRU_edge is not run
    edge_seq_len = seq_len-1
    Xin_egru1 = node_emb(Xin_egru1)
    Xin_egru2 = node_emb(Xin_egru2)
    Fto_in_egru = edge_emb(Fto_in_egru)
    Ffrom_in_egru = edge_emb(Ffrom_in_egru)
    ETD_input = torch.cat((Xin_egru1, Xin_egru2, Fto_in_egru, Ffrom_in_egru), 3)
    #print(ETD_input.shape, ETD_input[0, :10, :, 0])
    # copy GTE memory for ETD steps within each GTE step
    GTE_mem1 = GTE_mem1[0:params.max_num_node-1, :, :].permute(1, 0, 2)
    GTE_mem2 = GTE_mem2[0:params.max_num_node-1, :, :].permute(1, 0, 2)
    N, S, D = GTE_mem1.shape
    
    GTE_mem1 = GTE_mem1.unsqueeze(2)#.repeat(1, 1, S, 1)
    GTE_mem2 = GTE_mem2.unsqueeze(2)#.repeat(1, 1, S, 1)
    #print(GTE_mem1.shape, GTE_mem1[0, :10, :, 0])
    # merge 2nd dimension into batch dimension by packing
    ETD_input = pack_padded_sequence(ETD_input, edge_seq_len, batch_first=True, enforce_sorted=False).data.permute(1, 0, 2)
    GTE_mem1 = pack_padded_sequence(GTE_mem1, edge_seq_len, batch_first=True, enforce_sorted=False).data.permute(1, 0, 2)
    GTE_mem2 = pack_padded_sequence(GTE_mem2, edge_seq_len, batch_first=True, enforce_sorted=False).data.permute(1, 0, 2)
    Fto_out_egru = pack_padded_sequence(Fto_out_egru, edge_seq_len, batch_first=True, enforce_sorted=False).data.permute(1, 0)
    Ffrom_out_egru = pack_padded_sequence(Ffrom_out_egru, edge_seq_len, batch_first=True, enforce_sorted=False).data.permute(1, 0)
    
    Ffrom_pred = ETD1(ETD_input, GTE_mem1, tgt_mask=edge_mask)
    Fto_pred = ETD2(ETD_input, GTE_mem2, tgt_mask=edge_mask)
    # compute loss
    Fto_pred = Fto_pred.permute(0,2,1)
    Ffrom_pred = Ffrom_pred.permute(0,2,1)
    Fto_edge_loss = edge_CELoss(Fto_pred, Fto_out_egru)
    Ffrom_edge_loss = edge_CELoss(Ffrom_pred, Ffrom_out_egru)
    edge_loss = Fto_edge_loss + Ffrom_edge_loss
    edge_loss = torch.sum(edge_loss)/torch.sum(num_edges)

    return node_loss, edge_loss



def warmup_lr(params, epoch, batch_idx, lr_start, lr_end, num_epochs):
    total_num_steps = num_epochs*params.sample_batches
    current_step = epoch*params.sample_batches + batch_idx
    lr = lr_start + (lr_end-lr_start)*current_step/total_num_steps
    return lr

def train_epoch(params, config, writer, step, train_data, models, node_CELoss, edge_CELoss, 
                optimizer_node, scheduler_node, optimizer_edge, scheduler_edge, epoch):
    
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    # set to train mode
    if node_pred:
        mlp_node.train()
        gru_graph3.train()
    if edge_pred:
        gru_graph1.train()
        gru_graph2.train()
        gru_edge1.train()
        gru_edge2.train()   
    node_loss_avg = 0
    edge_loss_avg = 0
    loss_avg = 0
    warmup_epochs = 50
    node_mask = generate_square_subsequent_mask(params.max_num_node).to(DEVICE)
    edge_mask = generate_square_subsequent_mask(params.max_num_node-1).to(DEVICE)
    for batch_idx, batch in enumerate(train_data):
        # reset gradients
        if node_pred:
            mlp_node.zero_grad()
            gru_graph3.zero_grad()
        if edge_pred:
            gru_graph1.zero_grad()
            gru_graph2.zero_grad()
            gru_edge1.zero_grad()
            gru_edge2.zero_grad()
        # forward pass
        node_loss, edge_loss = forward_pass_model(params, config, batch, models, node_CELoss, edge_CELoss)
        # regularization
        l2_reg_edge = 0
        l2_reg_node = 0
        if node_pred:
            for param in mlp_node.parameters():
                l2_reg_node += torch.norm(param)
            for param in gru_graph3.parameters():
                l2_reg_node += torch.norm(param)
        if edge_pred:
            for param in gru_graph1.parameters():
                l2_reg_edge += torch.norm(param)
            for param in gru_graph2.parameters():
                l2_reg_edge += torch.norm(param)
            for param in gru_edge1.parameters():
                l2_reg_edge += torch.norm(param)
            for param in gru_edge2.parameters():
                l2_reg_edge += torch.norm(param)
        if node_pred:
            node_loss += params.reg*l2_reg_node
            # backward pass
            node_loss.backward()
            # take a step
            optimizer_node.step()
            if epoch<warmup_epochs:
                optimizer_node.param_groups[0]['lr'] = warmup_lr(params, epoch, batch_idx, params.node_lr_init, params.node_lr_init, warmup_epochs)
            else:
                scheduler_node.step()
            writer.add_scalar('Train loss: node', node_loss, global_step=step)
        else:
            node_loss = 0
        if edge_pred:
            edge_loss += params.reg*l2_reg_edge
            # backward pass
            edge_loss.backward()
            # take a step
            optimizer_edge.step()
            if epoch<warmup_epochs:
                optimizer_edge.param_groups[0]['lr'] = warmup_lr(params, epoch, batch_idx, params.node_lr_init, params.edge_lr_init, warmup_epochs)
            else:
                scheduler_node.step()
            writer.add_scalar('Train loss: edge', edge_loss, global_step=step)
        else:
            edge_loss = 0
        loss = node_loss + edge_loss
        #writer.add_scalar('Train loss: total', loss, global_step=step)
        loss_avg += loss.cpu().data.numpy()
        step+=1
    loss_avg = loss_avg/(batch_idx+1)
    models = node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2
    
    return loss_avg, step, models



def train(params, config, writer, graphs_train, models, hyperparam_str, class_weights):
    
    ordering, weighted_loss, node_pred, edge_pred, use_argmax, use_MHP = get_config(config)
    INV_NODE_WEIGHT, INV_EDGE_WEIGHT, SQRT_INV_NODE_WEIGHT, SQRT_INV_EDGE_WEIGHT = class_weights
    node_emb, edge_emb, mlp_node, gru_graph1, gru_graph2, gru_graph3, gru_edge1, gru_edge2 = models
    
    print('Training for '+ hyperparam_str)
    # initialize optimizer
    if node_pred:
        node_parameters=list(mlp_node.parameters())
        node_parameters.extend(list(gru_graph3.parameters()))
    if edge_pred:
        edge_parameters = list(gru_graph1.parameters())
        edge_parameters.extend(list(gru_edge1.parameters()))
        edge_parameters.extend(list(gru_graph2.parameters()))
        edge_parameters.extend(list(gru_edge2.parameters()))
    if node_pred:
        optimizer_node = optim.Adam(node_parameters, lr=params.node_lr)
        scheduler_node = StepLR(optimizer_node, step_size=params.node_step_decay_epochs, gamma=params.node_lr_rate)
    else:
        optimizer_node = None
        scheduler_node = None
    if edge_pred:
        optimizer_edge = optim.Adam(edge_parameters, lr=params.edge_lr)
        scheduler_edge = StepLR(optimizer_edge, step_size=params.edge_step_decay_epochs, gamma=params.edge_lr_rate)
    else:
        optimizer_edge = None
        scheduler_edge = None

    train_loss_all = list()
    # the outputs are padded with -1. Loss function doesnt compute loss corresponding to index -1
    if weighted_loss == 'inv':
        # weight the classes according to inverse of frequency of occurence
        node_CELoss = nn.CrossEntropyLoss(weight=INV_NODE_WEIGHT, ignore_index=-1, reduction='none')
        edge_CELoss = nn.CrossEntropyLoss(weight=INV_EDGE_WEIGHT, ignore_index=-1, reduction='none')
    elif weighted_loss == 'sqrt_inv':
        # weight the classes according to sqquare root of inverse of frequency of occurence
        node_CELoss = nn.CrossEntropyLoss(weight=SQRT_INV_NODE_WEIGHT, ignore_index=-1, reduction='none')
        SQRT_INV_EDGE_WEIGHT[params.num_edge_categories-3] = 0.2
        edge_CELoss = nn.CrossEntropyLoss(weight=SQRT_INV_EDGE_WEIGHT, ignore_index=-1, reduction='none')
    elif weighted_loss == 'no_edge_weighted':
        NO_NODE_ONLY = torch.ones(params.num_node_categories).to(DEVICE)
        NO_NODE_ONLY[params.num_node_categories-1] = 1
        node_CELoss = nn.CrossEntropyLoss(weight=NO_NODE_ONLY, ignore_index=-1, reduction='none')
        NO_EDGE_ONLY = torch.ones(params.num_edge_categories-2).to(DEVICE)
        NO_EDGE_ONLY[params.num_edge_categories-3] = 0.2
        edge_CELoss = nn.CrossEntropyLoss(weight=NO_EDGE_ONLY, ignore_index=-1, reduction='none')
    elif weighted_loss == 'none':
        node_CELoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
        edge_CELoss = nn.CrossEntropyLoss(ignore_index=-1, reduction='none')
    elif weighted_loss == 'inv_eff_num':
        node_weights = INV_NODE_WEIGHT
        edge_weights = INV_EDGE_WEIGHT
        node_CELoss = nn.CrossEntropyLoss(weight=node_weights, ignore_index=-1, reduction='none')
        edge_CELoss = nn.CrossEntropyLoss(weight=edge_weights, ignore_index=-1, reduction='none')

    nodeloss = node_CELoss
    edgeloss = edge_CELoss
    # Train
    epoch=1
    train_step=0
    val_step=0
    while epoch<=params.epochs:
        time_start = time.time()
        # train epoch
        train_loss, train_step, models = train_epoch(params, config, writer, train_step, graphs_train, models, nodeloss, edgeloss,
                                                    optimizer_node, scheduler_node, optimizer_edge, scheduler_edge, epoch)
        train_loss_all.append(train_loss)
        time_end = time.time()
        if node_pred:
            lr_node = optimizer_node.param_groups[0]['lr']
        else:
            lr_node = None
        if edge_pred:
            lr_edge = optimizer_edge.param_groups[0]['lr']
        else:
            lr_edge = 0
        print('Epoch: ', epoch, 'Training Loss: ', train_loss, 'node lr: ', lr_node, 'edge lr: ', lr_edge, 'time: ', time_end - time_start)
        epoch += 1

    # save models
    save_models(models, './models', hyperparam_str, config, params)
    # plot the result of training
    print('Plotting..')
    plot_loss(train_loss_all, 'Total '+ hyperparam_str)
    
    return models