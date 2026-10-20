import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import DenseGCNConv

class Graph:

    def __init__(self, num_nodes):
        if num_nodes != 14:
            raise ValueError('This skeleton requires 14 nodes.')
        self.num_nodes = num_nodes
        self.edges = [(0, 1), (0, 2), (0, 5), (1, 8), (1, 11), (2, 3), (3, 4), (5, 6), (6, 7), (8, 9), (9, 10), (11, 12), (12, 13)]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = torch.eye(self.num_nodes)
        for i, j in self.edges:
            A[i, j] = A[j, i] = 1.0
        return A

class LRC_Mask(nn.Module):

    def __init__(self, in_channels, num_nodes):
        super().__init__()
        self.edge_mask = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.node_mask = nn.Parameter(torch.zeros(1, in_channels, 1, num_nodes))

    def forward(self, x, A_static):
        x_masked = x * torch.sigmoid(self.node_mask)
        edge_weight = torch.sigmoid(self.edge_mask)
        edge_weight = (edge_weight + edge_weight.transpose(0, 1)) / 2.0
        I = torch.eye(self.edge_mask.size(0), device=x.device)
        edge_weight = edge_weight * (1 - I) + I
        A_evolved = A_static * edge_weight
        A_evolved = A_evolved / (A_evolved.sum(dim=-1, keepdim=True) + 1e-06)
        return (x_masked, A_evolved)

class TemporalTransformer(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.local_conv = nn.Conv2d(in_channels, out_channels, (5, 1), padding=(2, 0), stride=(stride, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=4, dim_feedforward=out_channels * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.local_conv(x)
        n, c, t, v = x.size()
        tokens = x.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)
        tokens = self.transformer(tokens)
        return tokens.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()

class STGCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropout, num_nodes):
        super().__init__()
        self.mask_module = LRC_Mask(in_channels, num_nodes)
        self.sgcn = DenseGCNConv(in_channels, out_channels)
        self.tcn = TemporalTransformer(out_channels, out_channels, stride, dropout)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.residual = nn.Conv2d(in_channels, out_channels, 1, (stride, 1)) if in_channels != out_channels or stride != 1 else nn.Identity()

    def forward(self, x, adj):
        n, c, t, v = x.size()
        res = self.residual(x)
        x_masked, adj_evolved = self.mask_module(x, adj)
        nodes = x_masked.permute(0, 2, 3, 1).contiguous().view(n * t, v, c)
        nodes = self.sgcn(nodes, adj_evolved).view(n, t, v, -1)
        nodes = nodes.permute(0, 3, 1, 2).contiguous()
        out = self.tcn(F.relu(nodes))
        return (F.relu(self.dropout(self.bn(out)) + res), adj_evolved)

class GRL(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return (grad_output.neg() * ctx.alpha, None)

class UDANetwork(nn.Module):

    def __init__(self, num_classes, in_channels, num_nodes, dropout):
        super().__init__()
        self.register_buffer('adj', Graph(num_nodes).A)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.st_gcn = nn.ModuleList([STGCN_Block(in_channels, 64, 1, dropout, num_nodes), STGCN_Block(64, 64, 2, dropout, num_nodes), STGCN_Block(64, 128, 1, dropout, num_nodes), STGCN_Block(128, 128, 2, dropout, num_nodes)])
        self.classifier = nn.Linear(128, num_classes)
        self.node_discriminator = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x, alpha=1.0, return_domain=False):
        if x.ndim == 5:
            if x.size(-1) != 1:
                raise ValueError('Only single-person five-dimensional input is supported.')
            x = x.squeeze(-1)
        if x.ndim != 4:
            raise ValueError('Expected [N,C,T,V] or [N,C,T,V,1].')
        n, c, t, v = x.size()
        x = self.data_bn(x.permute(0, 3, 1, 2).reshape(n, v * c, t))
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        last_adj = None
        for block in self.st_gcn:
            x, last_adj = block(x, self.adj)
        node_feat = x.mean(dim=2)
        feat = F.avg_pool2d(x, x.size()[2:]).view(n, -1)
        cls_out = self.classifier(feat)
        if return_domain:
            dom_out = self.node_discriminator(GRL.apply(node_feat, alpha))
            return (cls_out, dom_out, feat, last_adj)
        return cls_out
