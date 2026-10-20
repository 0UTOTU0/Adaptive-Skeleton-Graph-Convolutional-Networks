import torch
import torch.nn as nn
import torch.nn.functional as F

class Graph:

    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.edges = [(0, 1), (0, 2), (0, 5), (1, 0), (1, 8), (1, 11), (2, 0), (2, 3), (3, 2), (3, 4), (4, 3), (5, 0), (5, 6), (6, 5), (6, 7), (7, 6), (8, 1), (8, 9), (9, 8), (9, 10), (10, 9), (11, 1), (11, 12), (12, 11), (12, 13), (13, 12)]
        self.A = self.get_adjacency_matrix()

    def get_adjacency_matrix(self):
        A = torch.zeros((self.num_nodes, self.num_nodes), dtype=torch.float32)
        for i, j in self.edges:
            A[i, j] = 1.0
            A[j, i] = 1.0
        for i in range(self.num_nodes):
            A[i, i] = 1.0
        return A

class PureDenseGCNConv(nn.Module):

    def __init__(self, in_channels, out_channels, improved=False, bias=True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.improved = improved
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj, mask=None, add_loop=True):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        B, N, _ = adj.size()
        if add_loop:
            adj = adj.clone()
            idx = torch.arange(N, dtype=torch.long, device=adj.device)
            if self.improved:
                adj[:, idx, idx] = 2.0
            else:
                adj[:, idx, idx] = 1.0
        out = self.lin(x)
        deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)
        adj_norm = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
        out = torch.matmul(adj_norm, out)
        if self.bias is not None:
            out = out + self.bias
        if mask is not None:
            out = out * mask.view(B, N, 1).to(x.dtype)
        return out

class LRC_Mask(nn.Module):

    def __init__(self, in_channels, num_nodes):
        super().__init__()
        self.edge_mask = nn.Parameter(torch.zeros(num_nodes, num_nodes))
        self.node_mask = nn.Parameter(torch.zeros(1, in_channels, 1, num_nodes))

    def forward(self, x, A_static):
        node_weight = torch.sigmoid(self.node_mask)
        x_masked = x * node_weight
        edge_weight = torch.sigmoid(self.edge_mask)
        edge_weight = (edge_weight + edge_weight.transpose(0, 1)) / 2.0
        I = torch.eye(self.edge_mask.size(0), device=x.device, dtype=edge_weight.dtype)
        edge_weight = edge_weight * (1.0 - I) + I
        A_evolved = A_static * edge_weight
        row_sum = A_evolved.sum(dim=-1, keepdim=True) + 1e-06
        A_evolved = A_evolved / row_sum
        return (x_masked, A_evolved)

class TemporalTransformer(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropout):
        super().__init__()
        self.local_conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0), stride=(stride, 1))
        encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=4, dim_feedforward=out_channels * 4, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

    def forward(self, x):
        x = self.local_conv(x)
        n, c, t, v = x.size()
        x_trans = x.permute(0, 3, 2, 1).contiguous().view(n * v, t, c)
        x_trans = self.transformer(x_trans)
        x_trans = x_trans.view(n, v, t, c).permute(0, 3, 2, 1).contiguous()
        return x_trans

class STGCN_Block(nn.Module):

    def __init__(self, in_channels, out_channels, stride, dropout, num_nodes):
        super().__init__()
        self.mask_module = LRC_Mask(in_channels=in_channels, num_nodes=num_nodes)
        self.sgcn = PureDenseGCNConv(in_channels, out_channels)
        self.tcn = TemporalTransformer(out_channels, out_channels, stride=stride, dropout=dropout)
        self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(dropout)
        if in_channels != out_channels or stride != 1:
            self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1))
        else:
            self.residual = nn.Identity()

    def forward(self, x, adj):
        n, c, t, v = x.size()
        res = self.residual(x)
        x_masked, adj_evolved = self.mask_module(x, adj)
        x_sgcn = x_masked.permute(0, 2, 3, 1).contiguous().view(n * t, v, c)
        x_sgcn = self.sgcn(x_sgcn, adj_evolved)
        x_sgcn = x_sgcn.view(n, t, v, -1).permute(0, 3, 1, 2).contiguous()
        x_tcn = self.tcn(F.relu(x_sgcn))
        out = self.dropout(self.bn(x_tcn)) + res
        return (F.relu(out), adj_evolved)

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
        graph = Graph(num_nodes)
        self.register_buffer('adj', graph.A)
        self.data_bn = nn.BatchNorm1d(in_channels * num_nodes)
        self.st_gcn = nn.ModuleList([STGCN_Block(in_channels, 64, 1, dropout, num_nodes), STGCN_Block(64, 64, 2, dropout, num_nodes), STGCN_Block(64, 128, 1, dropout, num_nodes), STGCN_Block(128, 128, 2, dropout, num_nodes)])
        self.classifier = nn.Linear(128, num_classes)
        self.node_discriminator = nn.Sequential(nn.Conv1d(128, 64, 1), nn.ReLU(), nn.Conv1d(64, 1, 1))

    def forward(self, x, alpha=1.0, return_domain=False):
        if len(x.shape) == 5:
            x = x.squeeze(-1)
        n, c, t, v = x.size()
        x = x.permute(0, 3, 1, 2).reshape(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()
        last_adj = None
        for block in self.st_gcn:
            x, last_adj = block(x, self.adj)
        node_feat = x.mean(dim=2)
        feat = F.avg_pool2d(x, x.size()[2:]).view(n, -1)
        cls_out = self.classifier(feat)
        if return_domain:
            if alpha > 0.0:
                dom_out = self.node_discriminator(GRL.apply(node_feat, alpha))
            else:
                dom_out = torch.zeros(1, device=x.device)
            return (cls_out, dom_out, feat, last_adj)
        return cls_out
