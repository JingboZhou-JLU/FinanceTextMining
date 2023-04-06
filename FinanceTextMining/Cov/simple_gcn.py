import dgl.function as fn
import torch.nn as nn

class GCN(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCN, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g):
        # 使用DGL的内置函数来发送/聚合消息
        g.update_all(message_func=fn.copy_src(src='h', out='m'), reduce_func=fn.sum(msg='m', out='h'))
        # 计算新的节点特征
        h = self.linear(g.ndata['h'])
        return h



