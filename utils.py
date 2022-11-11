import numpy as np
import scipy.sparse as sp
import torch
import scipy.io as io
from scipy.sparse import csc_matrix
import os
import pandas as pd
from sklearn import preprocessing
import copy
import torch.nn as nn
from models import MLP


class CloseFormLoss(nn.Module):
    def __init__(self, DTI, fMRI):
        super(CloseFormLoss, self).__init__()
        self.tensorfMRI = fMRI
        self.tensorDTI = DTI

    def forward(self, param, labels):
        final_tensor = torch.concat((self.tensorfMRI, self.tensorDTI), dim=1)
        mlp = MLP(in_features=self.tensorDTI.shape[-1] * 2, hid_features=50, out_features=2)
        final_tensor = mlp(final_tensor)
        entropy = nn.CrossEntropyLoss()
        output1 = torch.mm(self.tensorfMRI, self.tensorfMRI.T) - torch.mm(self.tensorDTI, self.tensorDTI.T)
        output2 = torch.mm(self.tensorfMRI, self.tensorDTI.T)
        loss = (0.2 * torch.norm(output1, p=2) + 0.2 * torch.norm(output2, p=2) + 0.5 * entropy(final_tensor,
                                                                                                      labels))
        return loss


def node_feature_extraction(path1):
    la = True
    sym = True
    # robust = preprocessing.StandardScaler()
    if path1.endswith('.mat'):
        f = io.loadmat(path1)
        B = f['network']
        mat = csc_matrix(B).toarray()

    elif path1.endswith('.txt'):
        mat = np.loadtxt(path1, dtype=np.float32)

    mat1 = copy.deepcopy(mat)
    # print(f, type(f))
    mat = np.around(mat, 6)
    mat[np.isnan(mat)] = 0
    mat_t = mat.transpose()
    if not np.array_equal(mat, mat_t):
        return 'not symmetric, 行数：{}, 列数：{}'.format(mat.shape[0], mat.shape[1]), None, False, False
    row, column = mat.shape[0], mat.shape[1]
    data_matrix = []
    node_feature = []
    nonzero_neighbour = dict()

    for i in range(row):
        # calculate degree for each node. 1.degree 2.neighbour_index
        nonzero_neighbour[i] = [np.count_nonzero(mat[i, :]), np.where(mat[i, :] != 0)[0]]

        # using values to be the feature
        # data = [np.count_nonzero(mat[i, :]), np.min(mat[i, :]), np.max(mat[i, :]), np.mean(mat[i, :]),
        #         np.std(mat[i, :])]
        # data_matrix.append(data)
        # nonzero_neighbour[i] = np.where(mat[i, :] != 0)[0]

    data_matrix = np.array(data_matrix)
    # print("path: ", path1)
    # print("mat: ", mat)
    for i in range(row):
        nn = nonzero_neighbour[i][1]
        if len(nn) == 0:
            # print(path1)
            return i, False, False, True










        # using values to be the feature
        # node = [data_matrix[i][0], np.min(np.hstack((data_matrix[0:i, 1], data_matrix[i + 1:row, 1]))),
        #         np.max(np.hstack((data_matrix[0:i, 2], data_matrix[i + 1:row, 2]))),
        #         np.mean(np.hstack((data_matrix[0:i, 3], data_matrix[i + 1:row, 3]))),
        #         np.std(np.hstack((data_matrix[0:i, 4], data_matrix[i + 1:row, 4])))]
        # node = [data_matrix[i][0], np.min(data_matrix[nn, 1]) + 0.1,  # node_feature = [Deg(n), min(neighbour(n)), max(neighbour(n), mean((neighbour(n), std(neighbour(n))]
        #         np.max(data_matrix[nn, 2]),  # should be neighbour(Deg(n))
        #         np.mean(data_matrix[nn, 3]) + 0.1,
        #         np.std(data_matrix[nn, 4]) + 0.1]
        target = [nonzero_neighbour[nonzero_neighbour[i][1][k]][0] for k in range(len(nonzero_neighbour[i][1]))]
        node = [nonzero_neighbour[i][0], np.min(target),
                np.max(target),
                np.mean(target) if np.mean(target) != 0 else 1,
                np.std(target) if np.std(target) != 0 else 1
                ]
        node_feature.append(node)
        print("path: ", path1)
        print('node_feature: ', node, '\n')
    node_feature = np.array(node_feature)
    mat[mat == 0] = 0
    node_feature = normalize_features(node_feature)
    return mat, node_feature, True, sym


def encode_onehot(labels):
    # The classes must be sorted before encoding to enable static class encoding.
    # In other words, make sure the first class always maps to index 0.
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot


def load_data(path="./data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels = encode_onehot(idx_features_labels[:, -1])

    # build graph
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    print('outputs: ', output)
    preds = output.max(1)[1].type_as(labels)
    print('preds: ', preds)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    print(correct, len(labels))
    return correct / len(labels)


def extract_labels(path, usecols):
    file = pd.read_csv(path, usecols=usecols)
    gender = {'M':1,'F':0}
    file['Gender'] = file['Gender'].map(gender)
    matrix = np.array(file)
    new_dict = {}
    for i in matrix:
        new_dict[i[0]] = i[1]
    return new_dict


def matrix_extraction():
    # load data
    path = '/Users/norahc/Downloads/QCed_fMRInetwork'
    path2 = '/Users/norahc/Downloads/QCed_DTInetwork'
    path3 = '/Users/norahc/Downloads/hcp_annontations.csv'
    usecols = ['Subject', 'Gender']
    not_used = []
    tensor_mat, tensor_mat_DT = torch.zeros((1, 82, 82)), torch.zeros((1, 82, 82))
    tensor_nf, tensor_nf_DT = torch.zeros((1, 82, 5)), torch.zeros((1, 82, 5))
    gender_labels = torch.LongTensor(1028).zero_()
    i = 0
    total = 0
    ne = 0
    labels = extract_labels(path3, usecols=usecols)
    with open('no_neighbours.txt', 'w', encoding='utf-8') as d:
        for root, dir, files in os.walk(path):
            for file in files:
                if not (file.endswith('.mat') or file.endswith('.txt')):
                    continue
                total += 1
                path1 = os.path.join(root, file)
                file_type =  '_fMRI'
                mat, node_feature, la, sym = node_feature_extraction(path1)
                if not la:
                    ne += 1
                    if not sym:
                        ne += 1
                        d.write("文件名: {}, {}, network: {}\n".format(file, mat, file_type))
                    else:
                        d.write("文件名: {}, 行数：{}, network: {}\n".format(file, mat, file_type))
                    continue
                subject = int(file[:6])
                l = labels[subject]
                gender_labels[i] = bool(l)
                i += 1
                if mat.shape[0] != mat.shape[1]:
                    not_used.append(file)
                    continue
                f_DT_1 = file[:-4] + '_density.txt'
                f_DT = os.path.join(path2, f_DT_1)
                mat_DT, node_feature_DT, la, sym = node_feature_extraction(f_DT)
                file_type = '_DTI'
                if not la:
                    ne += 1
                    if not sym:
                        d.write("文件名: {}, {}, network: {}\n".format(f_DT_1, mat_DT, file_type))
                    else:
                        d.write("文件名: {}, 行数：{}, network: {}\n".format(f_DT_1, mat_DT, file_type))
                    continue
                if mat_DT.shape[0] != mat.shape[1]:
                    not_used.append(f_DT)
                    continue
                mat, node_features = torch.from_numpy(mat).float(), torch.from_numpy(node_feature).float()
                tensor_mat, tensor_nf = torch.cat((tensor_mat, mat.clone().view(1, 82, 82)), 0), torch.cat(
                    (tensor_nf, node_features.clone().view(1, 82, 5)), 0)

                mat_DT, node_features_DT = torch.from_numpy(mat_DT).float().view(1, 82, 82), torch.from_numpy(
                    node_feature_DT).float().view(1, 82, 5)
                tensor_mat_DT, tensor_nf_DT = torch.cat((tensor_mat_DT, mat_DT), 0), torch.cat(
                    (tensor_nf_DT, node_features_DT), 0)
        mat, node_features = tensor_mat[1:, :, :], tensor_nf[1:, :, :]
        mat_DT, node_features_DT = tensor_mat_DT[1:, :, :], tensor_nf_DT[1:, :, :]




    print(mat.shape, node_features.shape)
    return mat, mat_DT, node_features, node_features_DT, gender_labels




