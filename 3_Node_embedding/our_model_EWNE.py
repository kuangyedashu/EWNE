from __future__ import print_function, division
import argparse
import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.preprocessing import normalize
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear
from torch_geometric.nn import GATConv
from utils import load_data, load_graph, get_core, cosine_similarity, naeval
from GNN import GraphAttentionLayer
# from evaluation import eva
import copy

class FirstWeight(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super(FirstWeight, self).__init__()
        self.conv1 = GraphAttentionLayer(in_dim, out_dim, 0.2)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = torch.tanh(h)
        return h

class HighWeight(torch.nn.Module):
    def __init__(self, in_dim, hidden):
        super(HighWeight, self).__init__()
        self.lstm = torch.nn.LSTM(in_dim, hidden, 1, bias=False)

    def forward(self, x, h, c):
        x, (h, c) = self.lstm(x, (h, c))
        return x, (h, c)
class EWNELayer(torch.nn.Module):
    def __init__(self, in_dim):
        super(EWNELayer, self).__init__()
        self.FirstWeight_func = FirstWeight(in_dim, dim)
        self.HighWeight_func = HighWeight(dim, lstm_hidden)

    def forward(self, x, edge_index, h, c):
        x = self.FirstWeight_func(x, edge_index, 0.2)
        x = x[None, :]
        x, (h, c) = self.HighWeight_func(x, h, c)
        x = x[0]
        return x, (h, c)

class EWNE(torch.nn.Module):#not lazy:串联模式
    def __init__(self, in_dim, out_dim, device):
        super(EWNE, self).__init__()
        self.device = device
        self.lin1 = torch.nn.Linear(in_dim, dim)
        self.gplayers = torch.nn.ModuleList([EWNELayer(dim) for i in range(layer_num)])
        self.lin2 = torch.nn.Linear(dim, out_dim)
    def forward(self, x, edge_index):
        x = self.lin1(x)
        h = torch.zeros(1, x.shape[0], lstm_hidden).to(self.device)
        c = torch.zeros(1, x.shape[0], lstm_hidden).to(self.device)
        for i, l in enumerate(self.gplayers):
            x, (h, c) = self.gplayers[i](x, edge_index, h, c)
        x = self.lin2(x)
        h = x
        z = F.normalize(h, p=2, dim=1)
        A_pred = dot_product_decode(z)
        return A_pred,z,h
# class EWNE(nn.Module):#并联模式
#     def __init__(self, num_features, hidden_size, embedding_size, alpha):
#         super(EWNE, self).__init__()
#         self.hidden_size = hidden_size
#         self.embedding_size = embedding_size
#         self.alpha = alpha
#         self.conv1 = GraphAttentionLayer(num_features, embedding_size, alpha)

#     def forward(self, x, adj, M):
#         h = self.conv1(x, adj, M)
#         z = F.normalize(h, p=2, dim=1)
#         A_pred = dot_product_decode(z)
#         return A_pred,z,h


def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred


dim = 256   #这里是GAT的隐层神经元
lstm_hidden = 256  #这里是LSTM的隐层神经元
heads = 1
layer_num = 1

def pretrain_daegc(dataset):#是程序开始训练的入口了
    model = EWNE(143, 32, device).to(device)#第一个参数是输入维度（这个输入维度，可能需要改）#第二个参数，就是输出维度！
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    #Some porcess
    adj, adj_label = load_graph(args.name, args.k)
    adj_dense = adj.to_dense()
    adj_numpy = adj_dense.data.cpu().numpy()
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    M = torch.Tensor(M_numpy).cuda()

    adj = adj_dense.cuda()
    adj_label = adj_label.cuda()

    # cluster parameter initiate
    data = torch.Tensor(dataset.x).cuda()
    y = dataset.y

    for epoch in range(30):#之前是30
        model.train()
        A_pred,z,h = model(data, adj)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _,z,h = model(data, adj)
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(z.data.cpu().numpy())
            print('这是训练第' + str(epoch) + '次的训练的结果（此处结果不重要，我们要的是学习到的隐层特征）')
            # eva(y, kmeans.labels_, epoch)
            if epoch == 28:
                print('接下来把训练数次得到的特征h保存到变量laten_feature中并写入文件')
                h = h.cpu().numpy()
                h = h.tolist()
                laten_feature = h
                file.write(str(laten_feature))
        # torch.save(model.state_dict(), 'predaegc_ppi.pkl')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='train',formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='krogan2006core')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--n_clusters', default=6, type=int)
    # parser.add_argument('--hidden1_dim', default=256, type=int)
    # parser.add_argument('--hidden2_dim', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    dataset = load_data(args.name)
    if args.name == 'collins':
        args.lr = 0.001
        args.k = None
        args.n_clusters = 34    
        args.input_dim = 141
    if args.name == 'gavin':
        args.lr = 0.001
        args.k = None
        args.n_clusters = 12
        args.input_dim = 283#这个数
    if args.name == 'krogan2006core':
        args.lr = 0.001
        args.k = None
        args.n_clusters = 16
        args.input_dim = 287


    print(args)
    file=open("参数敏感性实验得到的数据/网络层数的影响/krogan2006cora1_laten_feature.txt","w")#用于保存得到的特征向量
    pretrain_daegc(dataset)