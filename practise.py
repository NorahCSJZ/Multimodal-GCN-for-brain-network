import pandas as pd
import numpy as np
import torch
import scipy.io as io
from scipy.sparse import csc_matrix

path1 = '/Users/norahc/Downloads/QCed_fMRInetwork/687163.mat'
f = io.loadmat(path1)
B = f['network']
mat = csc_matrix(B).toarray()
mat = np.around(mat, 6)
mat[np.isnan(mat)] = 0

row, column = mat.shape[0], mat.shape[1]
data_matrix = []
nonzero_neighbour = dict()
for i in range(row):
    data = [np.count_nonzero(mat[i, :]), np.min(mat[i, :]), np.max(mat[i, :]), np.mean(mat[i, :]),
            np.std(mat[i, :])]
    data_matrix.append(data)
    nonzero_neighbour[i] = np.where(mat[i, :] != 0)[0]
data_matrix = np.array(data_matrix)
for i in range(row):
    nn = nonzero_neighbour[i]
    if len(nn) == 0:
        print(1)
print(mat)