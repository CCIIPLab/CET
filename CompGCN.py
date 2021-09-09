import torch
import torch.nn as nn
import dgl.function as fn


class CompGraphConv(nn.Module):
    def __init__(self, in_dim, out_dim, comp_fn='sub', batchnorm=False, dropout=0.1, self_loop=False, activation=None):
        super(CompGraphConv, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.comp_fn = comp_fn
        self.actvation = activation
        self.batchnorm = batchnorm
        self.self_loop = self_loop

        # define dropout layer
        self.dropout = nn.Dropout(dropout)

        # define batch norm layer
        if self.batchnorm:
            self.bn = nn.BatchNorm1d(out_dim)

        # defnie in/out/loop transform layer
        self.W_O = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.W_O.weight, gain=nn.init.calculate_gain('relu'))
        self.W_I = nn.Linear(self.in_dim, self.out_dim)
        nn.init.xavier_uniform_(self.W_I.weight, gain=nn.init.calculate_gain('relu'))
        if self_loop:
            self.W_S = nn.Linear(self.in_dim, self.out_dim)
            # self loop embedding
            self.loop_rel = nn.Parameter(torch.Tensor(1, self.in_dim))
            nn.init.xavier_normal_(self.loop_rel)

        # define relation transform layer
        self.W_R = nn.Linear(self.in_dim, self.out_dim)

    def ccorr(self, a, b):
        def com_mult(a, b):
            r1, i1 = a[..., 0], a[..., 1]
            r2, i2 = b[..., 0], b[..., 1]
            return torch.stack([r1 * r2 - i1 * i2, r1 * i2 + i1 * r2], dim=-1)

        def conj(a):
            a[..., 1] = -a[..., 1]
            return a

        return torch.irfft(com_mult(conj(torch.rfft(a, 1)), torch.rfft(b, 1)), 1, signal_sizes=(a.shape[-1],))

    def forward(self, g, n_feats, r_feats, num_rels):
        with g.local_scope():
            g.srcdata['h'] = n_feats
            if self.self_loop:
                r_feats = torch.cat((r_feats, self.loop_rel), 0)
            else:
                r_feats = r_feats
            g.edata['h'] = r_feats[g.edata['etype']]

            if self.comp_fn == 'sub':
                g.apply_edges(fn.u_sub_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'mul':
                g.apply_edges(fn.u_mul_e('h', 'h', out='comp_h'))
            elif self.comp_fn == 'ccorr':
                g.apply_edges(lambda edges: {'comp_h': self.ccorr(edges.src['h'], edges.data['h'])})
            else:
                raise ValueError('comp_fn %s is not supported' % self.comp_fn)

            comp_h = g.edata['comp_h']

            new_comp_h = torch.where((g.edata['etype'] < num_rels // 2).unsqueeze(1).repeat(1, self.out_dim), self.W_I(comp_h),
                                     self.W_O(comp_h))

            g.edata['new_comp_h'] = new_comp_h

            g.update_all(fn.copy_e('new_comp_h', 'm'), fn.mean('m', 'comp_edge'))

            n_out_feats = self.dropout(g.dstdata['comp_edge'])

            if self.self_loop:
                dst_feats = n_feats[:g.number_of_dst_nodes()]
                comp_h_s = self.ccorr(dst_feats, r_feats[-1])
                n_out_feats += self.W_S(comp_h_s)

            r_out_feats = self.W_R(r_feats)

            if self.batchnorm:
                n_out_feats = self.bn(n_out_feats)

            if self.actvation is not None:
                n_out_feats = self.actvation(n_out_feats)

            if self.self_loop:
                return n_out_feats, r_out_feats[:-1]
            else:
                return n_out_feats, r_out_feats


class CompGCN(nn.Module):
    def __init__(self,
                 num_nodes,
                 num_rels,
                 num_types,
                 embed_dim,
                 num_bases=-1,
                 dropout=0.3,
                 num_layers=1,
                 activation='none',
                 self_loop=False):
        super(CompGCN, self).__init__()
        self.num_nodes = num_nodes
        self.num_rels = num_rels
        self.num_types = num_types
        self.embed_dim = embed_dim
        self.num_bases = num_bases
        if activation == 'none':
            activation_fn = None
        elif activation == 'relu':
            activation_fn = torch.relu
        elif activation == 'tanh':
            activation_fn = torch.tanh
        else:
            raise ValueError('%s is not supported' % activation)

        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(
                CompGraphConv(embed_dim, embed_dim, activation=activation_fn, self_loop=self_loop)
            )

        # Initial relation embeddings
        if num_bases > 0:
            self.basis = nn.Parameter(torch.Tensor(num_bases, embed_dim))
            self.weight = nn.Parameter(torch.Tensor(num_rels, num_bases))
            nn.init.uniform_(self.basis, a=-10 / embed_dim, b=10 / embed_dim)
            nn.init.xavier_normal_(self.basis)
        else:
            self.rel_embeds = nn.Parameter(torch.randn(num_rels, embed_dim))
            nn.init.uniform_(self.rel_embeds, a=-10/embed_dim, b=10/embed_dim)
        # Node embeddings
        self.entity = nn.Parameter(torch.randn(num_nodes, embed_dim))
        nn.init.uniform_(self.entity, a=-10/embed_dim, b=10/embed_dim)

        # Dropout
        self.dropouts = nn.ModuleList()
        for i in range(num_layers):
            self.dropouts.append(
                nn.Dropout(dropout)
            )

        self.fc = nn.Linear(embed_dim, num_types)

    def forward(self, blocks):
        # node and relation features
        n_feats = torch.index_select(self.entity, 0, index=blocks[0].srcdata['id'])
        if self.num_bases > 0:
            r_feats = torch.mm(self.weight, self.bias)
        else:
            r_feats = self.rel_embeds

        # neighbor aggregation
        for graph, layer, dropout in zip(blocks, self.layers, self.dropouts):
            n_feats, r_feats = layer(graph, n_feats, r_feats, self.num_rels)
            n_feats = dropout(n_feats)

        # classification
        predict = self.fc(n_feats).sigmoid()

        return predict
