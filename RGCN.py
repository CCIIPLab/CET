import torch
import torch.nn as nn
import dgl.function as fn

class RGCN(nn.Module):
    def __init__(self, embedding_dim, num_nodes, num_rels, num_types,
                 self_loop=False, regularizer='basis', num_bases=-1, num_layers=1, activation='none'):
        super(RGCN, self).__init__()
        self.num_rels = num_rels

        self.conv_layers = nn.ModuleList()
        if activation=='none':
            activation_fn = None
        elif activation == 'relu':
            activation_fn = torch.relu
        elif activation == 'tanh':
            activation_fn = torch.tanh
        else:
            raise ValueError('%s is not supported' % activation)

        for i in range(num_layers):
            self.conv_layers.append(
                RGCNLayer(embedding_dim, embedding_dim, 2*num_rels, self_loop=self_loop,
                          regularizer=regularizer, num_bases=num_bases, activation=activation_fn)
            )

        self.fc = nn.Linear(embedding_dim, num_types)

        self.entity = nn.Parameter(torch.randn(num_nodes, embedding_dim))
        nn.init.uniform_(self.entity, a=-10/embedding_dim, b=10/embedding_dim)

        self.device = torch.device('cuda')

    def forward(self, blocks):
        h = torch.index_select(self.entity, 0, blocks[0].srcdata['id'])

        for layer, block in zip(self.conv_layers, blocks):
            h = layer(block, h)

        output = torch.sigmoid(self.fc(h))
        return output


class RGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rels, self_loop=False, regularizer="basis", num_bases=-1, activation=None):
        super(RGCNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.self_loop = self_loop
        self.num_bases = num_bases
        self.activation = activation

        if regularizer == "basis":
            if num_bases == -1:
                self.weight = nn.Parameter(torch.randn(num_rels, in_dim, out_dim))
            else:
                self.basis = nn.Parameter(torch.randn(num_bases, in_dim * out_dim))
                self.coef = nn.Parameter(torch.randn(num_rels, num_bases))
            self.message_func = self.bias_message_func
        elif regularizer == "bdd":
            if in_dim % num_bases != 0 and out_dim % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases (%d).' % self.num_bases)
            self.submat_in = in_dim // self.num_bases
            self.submat_out = out_dim // self.num_bases

            self.weight = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            self.message_func = self.bdd_message_func
        else:
            raise ValueError('Regularizer must be either "basis" or "bdd"')

        if self.self_loop:
            self.self_loop_weight = nn.Parameter(torch.randn(in_dim, out_dim))

    def bias_message_func(self, edges):
        src = edges.src['h']
        if self.num_bases == -1:
            weight = self.weight
        else:
            weight = torch.matmul(self.coef, self.basis)
            weight = weight.reshape(self.num_rels, self.in_dim, self.out_dim)

        w = weight[edges.data['etype']]
        msg = torch.bmm(src.unsqueeze(1), w).squeeze(1)
        return {'msg': msg}

    def bdd_message_func(self, edges):
        src = edges.src['h'].view(-1, 1, self.submat_in)
        weight = self.weight[edges.data['etype']].view(-1, self.submat_in, self.submat_out)
        msg = torch.bmm(src, weight).view(-1, self.out_dim)
        return {'msg': msg}


    def forward(self, graph, in_feat):
        with graph.local_scope():
            graph.srcdata['h'] = in_feat
            graph.update_all(self.message_func, fn.mean('msg', 'h'))
            output = graph.dstdata['h']

            if self.self_loop:
                dst_feat = in_feat[:graph.number_of_dst_nodes()]
                output += torch.matmul(dst_feat, self.self_loop_weight)

            if self.activation:
                output = self.activation(output)

            return output


