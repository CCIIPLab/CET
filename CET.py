import torch
import torch.nn as nn
import dgl.function as fn


class CET(nn.Module):
    def __init__(self, args, num_nodes, num_rels, num_types):
        super(CET, self).__init__()
        self.embedding_dim = args['hidden_dim']
        self.embedding_range = 10 / self.embedding_dim
        self.num_rels = num_rels

        self.layer = CETLayer(num_nodes, self.embedding_dim, num_types, args['temperature'])

        self.entity = nn.Parameter(torch.randn(num_nodes, self.embedding_dim))
        nn.init.uniform_(tensor=self.entity, a=-self.embedding_range, b=self.embedding_range)

        self.relation = nn.Parameter(torch.randn(num_rels, self.embedding_dim))
        nn.init.uniform_(tensor=self.relation, a=-self.embedding_range, b=self.embedding_range)

        self.device = torch.device('cuda')

    def forward(self, blocks):
        src = torch.index_select(self.entity, 0, blocks[0].srcdata['id'])

        # get edge embeddings, etype > self.num_rels means the inverse relation
        etype = blocks[0].edata['etype']
        relations = torch.index_select(self.relation, 0, etype % self.num_rels)
        relations[etype >= self.num_rels] = relations[etype >= self.num_rels] * -1

        output = self.layer(blocks[0], src, relations)

        return output


class CETLayer(nn.Module):
    def __init__(self, num_nodes, embedding_dim, num_types, temperature):
        super(CETLayer, self).__init__()
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.num_types = num_types
        self.fc = nn.Linear(embedding_dim, num_types)
        self.temperature = temperature
        self.test = False

    def reduce_func(self, nodes):
        # neighbors -> type
        msg = torch.relu(nodes.mailbox['msg'])
        predict1 = self.fc(msg)

        # mean pool agg -> type
        agg = nodes.mailbox['msg'].mean(1, keepdim=True)
        predict2 = self.fc(torch.relu(agg))

        predict = torch.cat([predict1, predict2], dim=1)

        weight = torch.softmax(self.temperature * predict, dim=1)
        predict = (predict * weight.detach()).sum(1).sigmoid()

        return {'predict': predict}

    def forward(self, graph, src_embedding, edge_embedding):
        assert len(edge_embedding) == graph.num_edges(), print('every edge should have a type')
        with graph.local_scope():
            graph.srcdata['h'] = src_embedding
            graph.edata['h'] = edge_embedding

            # message passing
            graph.update_all(fn.u_add_e('h', 'h', 'msg'), self.reduce_func)
            return graph.dstdata['predict']

