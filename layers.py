import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch_geometric.nn import  SAGEConv


class GINConv(nn.Module):
    def __init__(self, apply_func, init_eps=0, learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        if learn_eps:
            self.eps = nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, adj, feat):
        neigh_feat = torch.spmm(adj, feat)
        feat_f = (1 + self.eps) * feat + neigh_feat
        if self.apply_func is not None:
            feat_f = self.apply_func(feat_f)
        return feat_f

class GIN(nn.Module):
    def __init__(self, in_feats, hidden_size, num_classes,device):
        super(GIN, self).__init__()
        line_layer1 = nn.Linear(in_feats, hidden_size)
        line_layer2 = nn.Linear(hidden_size, num_classes)
        self.conv1 = GINConv(line_layer1)
        self.conv2 = GINConv(line_layer2)
        self.device = device

    def forward(self, feats ,adj):
        h = self.conv1(adj, feats)
        h = F.relu(h)
        h = self.conv2(adj, h)
        return h

# GraphSAGE代码
class GraphSAGE(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout, device):
        super(GraphSAGE, self).__init__()
        self.sage1 = SAGEConv(input_dim, hidden_dim)
        self.sage2 = SAGEConv(hidden_dim, output_dim)
        self.dropout = dropout
        self.batchnorm = nn.BatchNorm1d(output_dim)    # Add BatchNormalization Layer
        self.device = device

    def forward(self, x, adj):
        # Convert adjacency matrix to edge index
        edge_index = adj.nonzero(as_tuple=False).t().contiguous()

        x_sage = F.dropout(x, self.dropout, training=self.training)
        x_sage = F.relu(self.sage1(x_sage, edge_index))
        x_sage = F.dropout(x_sage, self.dropout, training=self.training)
        x_sage = F.relu(self.sage2(x_sage, edge_index))


        x = self.batchnorm(x_sage)
        return x



#GAT代码
class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903. This part of code refers to the implementation of https://github.com/Diego999/pyGAT.git

    """

    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h, adj):
        Wh = torch.mm(h, self.W)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        a_input = self._prepare_attentional_mechanism_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(2))  # （N，N）

        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)

        h_prime = torch.matmul(attention, Wh)
        return F.relu(h_prime)

    def _prepare_attentional_mechanism_input(self, Wh):
        N = Wh.size()[0]  # number of nodes

        # Below, two matrices are created that contain embeddings in their rows in different orders.
        # (e stands for embedding)
        # These are the rows of the first matrix (Wh_repeated_in_chunks):
        # e1, e1, ..., e1,            e2, e2, ..., e2,            ..., eN, eN, ..., eN
        # '-------------' -> N times  '-------------' -> N times       '-------------' -> N times
        #
        # These are the rows of the second matrix (Wh_repeated_alternating):
        # e1, e2, ..., eN, e1, e2, ..., eN, ..., e1, e2, ..., eN
        # '----------------------------------------------------' -> N times
        #

        Wh_repeated_in_chunks = Wh.repeat_interleave(N, dim=0)
        Wh_repeated_alternating = Wh.repeat(N, 1)
        # Wh_repeated_in_chunks.shape == Wh_repeated_alternating.shape == (N * N, out_features)

        # The all_combination_matrix, created below, will look like this (|| denotes concatenation):
        # e1 || e1
        # e1 || e2
        # e1 || e3
        # ...
        # e1 || eN
        # e2 || e1
        # e2 || e2
        # e2 || e3
        # ...
        # e2 || eN
        # ...
        # eN || e1
        # eN || e2
        # eN || e3
        # ...
        # eN || eN

        all_combinations_matrix = torch.cat([Wh_repeated_in_chunks, Wh_repeated_alternating], dim=1)
        # all_combinations_matrix.shape == (N * N, 2 * out_features)

        return all_combinations_matrix.view(N, N, 2 * self.out_features)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'

#自注意力机制
class selfattention(nn.Module):
    def __init__(self, sample_size, d_k, d_v):
        super().__init__()
        self.d_k = d_k
        self.d_v = d_v
        self.query = nn.Linear(sample_size, d_k)
        self.key = nn.Linear(sample_size, d_k)
        self.value = nn.Linear(sample_size, d_v)

    def forward(self, x):
        x =x.T
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        att = torch.matmul(q, k.transpose(0, 1)) / np.sqrt(self.d_k)
        att = torch.softmax(att, dim=1)
        output = torch.matmul(att, v)
        return output.T

#GCN代码
class GraphConvolution(nn.Module):

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'



#GAC代码
class GraphAttentionLayer_GAC(nn.Module):
    def __init__(self, in_features, out_features, alpha=0.2):
        super(GraphAttentionLayer_GAC, self).__init__()
        self.alpha = alpha
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, input, adj):
        h = torch.matmul(input, self.W)
        N = h.size()[0]
        a_input = torch.cat([h.repeat(1, N).view(N*N, -1), h.repeat(N, 1)], dim=1).view(N, -1, 2*h.size(1))
        e = F.leaky_relu(torch.matmul(a_input, self.a).squeeze(2))
        attention = F.softmax(e, dim=1)
        h_prime = torch.matmul(attention, h)
        return F.elu(h_prime)

class GAC(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GAC, self).__init__()
        self.gc1 = GraphAttentionLayer_GAC(input_dim, hidden_dim)
        self.gc2 = GraphAttentionLayer_GAC(hidden_dim, output_dim)

    def forward(self, input, adj):
        x = F.relu(self.gc1(input, adj))
        x = self.gc2(x, adj)
        return x





