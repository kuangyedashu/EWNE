import numpy as np
import scipy.sparse as sp
import h5py
import torch
from torch.utils.data import Dataset
import networkx as nx


def load_graph(dataset, k):
    if k:
        path = 'graph/{}{}_graph.txt'.format(dataset, k) 
    else:
        path = 'graph/{}_graph.txt'.format(dataset) 

    data = np.loadtxt('data/{}.txt'.format(dataset))
    n, _ = data.shape

    idx = np.array([i for i in range(n)], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt(path, dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(n, n), dtype=np.float32)

    # build symmetric adjacency matrix
    adj_noeye = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj_noeye + sp.eye(adj_noeye.shape[0])
    adj = normalize(adj)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    adj_label = sparse_mx_to_torch_sparse_tensor(adj_noeye)

    return adj,adj_label


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class load_data(Dataset):
    def __init__(self, dataset):
        self.x = np.loadtxt('data/{}.txt'.format(dataset), dtype=float)
        self.y = np.loadtxt('data/{}_label.txt'.format(dataset), dtype=int)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return torch.from_numpy(np.array(self.x[idx])),\
               torch.from_numpy(np.array(self.y[idx])),\
               torch.from_numpy(np.array(idx))


def get_core():
    G = nx.Graph()
    edges = []
    file = open('graph/ppi_graph.txt', 'r')
    for line in file.readlines():
        line = line.strip().split(' ')
        tmp = (int(line[0]), int(line[1]))
        edges.append(tmp)
    G.add_edges_from(edges)
    cliques = nx.find_cliques(G)
    core_cs = sorted(cliques, key=lambda item: -len(item))
    res = []
    for core_c in core_cs:
        if len(core_c) > 2:
            res.append(core_c)
    return res


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)
    dot_product, square_sum_x, square_sum_y = 0, 0, 0
    for i in range(len(x)):
        dot_product += x[i] * y[i]
        square_sum_x += x[i] * x[i]
        square_sum_y += y[i] * y[i]
    cos = dot_product / (np.sqrt(square_sum_x) * np.sqrt(square_sum_y))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内
# a = [0, 1, 1, 1, 2, 1, 0, 2, 1, 1]
# b = [1, 0, 1, 1, 2, 1, 1, 1, 1, 0]
# print(cosine_similarity(a, b))


#定义一个邻域亲和评分：NA，用于检测两个模块之间的评分是否大于等于0.2
def naeval(a,b):
    temp = [i for i in a if i in b]
    shang = len(temp) * len(temp)
    xia = len(a) * len(b)
    res = shang / xia
    if res > 0:
        return 1
    else:
        return 0


