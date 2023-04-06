import dgl.function as fn
import torch.nn as nn
import dgl

class GCNConv(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNConv, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature, edge_weight):
        with g.local_scope():
            # set the node feature
            g.ndata['h'] = feature.to_dense()
            # set the edge weight
            g.edata['w'] = edge_weight

            # update node feature with message passing
            g.update_all(message_func=fn.u_mul_e('h', 'w', 'm'), reduce_func=fn.sum('m', 'h'))

            # get the updated node feature
            h = g.ndata.pop('h')

            # apply linear transformation to get output
            out = self.linear(h)

        return out

