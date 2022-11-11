from __future__ import division
from __future__ import print_function

import os
import glob
import time
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import math
import torch.optim.lr_scheduler as lr_scheduler

from utils import load_data, accuracy, matrix_extraction, CloseFormLoss
from utils import node_feature_extraction as nfe
from utils import extract_labels
from models import GAT, SpGAT, MLP
from my_dataset import MyDatasets

# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
parser.add_argument('--sparse', action='store_true', default=False, help='GAT with sparse version or not.')
parser.add_argument('--seed', type=int, default=72, help='Random seed.')
parser.add_argument('--epochs', type=int, default=1224, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, default=5e-5, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
parser.add_argument('--dropout', type=float, default=0.6, help='Dropout rate (1 - keep probability).')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=10, help='Patience')
parser.add_argument('--nout', type=int, default=10, help='Patience')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

mat, mat_DT, node_features, node_features_DT, labels = matrix_extraction()

# Model and optimizer
if args.sparse:
    model = SpGAT(nfeat=node_features.shape[2],
                nhid=args.hidden,
                nclass=args.nout,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

else:
    model = GAT(nfeat=node_features.shape[2],
                nhid=args.hidden,
                nclass=args.nout,
                dropout=args.dropout,
                nheads=args.nb_heads,
                alpha=args.alpha)

idx_train = range(600)  # train batch
idx_val = range(600, 800)  # validation batch
idx_test = range(800, 953)  # test batch
mlp = MLP(in_features=args.nout * 2, hid_features=args.hidden, out_features=2)

# params = [
        #             {
        #                 "params": [value],
        #                 "name": key,
        #                 "param_size": value.size(),
        #                 "nelement": value.nelement(),
        #             }
        #             for key, value in self.model.named_parameters()
        #         ]
        #
        #         a = self.criterion_auc.a
        #         b = self.criterion_auc.b
        #         alpha = self.criterion_auc.alpha
        #
        #         params.append({"params": a,
        #                        "name": "primal_a",
        #                        "param_size": a.size(),
        #                        "nelement": a.nelement()})
parm = [
    {
        "params": [value],
        "name": key,
        "param size": value.size(),
        "nelement": value.nelement(),
    }
    for key, value in model.named_parameters()
]

for key, value in mlp.named_parameters():
    parm.append(
        {
            "params": [value],
            "name": key,
            "param size": value.size(),
            "nelement": value.nelement(),
        }
    )

# optimizer
optimizer = optim.Adam(parm, lr=args.lr, weight_decay=args.weight_decay)


entropy = torch.nn.CrossEntropyLoss()

if args.cuda:
    model.cuda()
    features = node_features.cuda()
    adj = mat.cuda()

    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()

# mat, mat_DT, node_features, node_features_DT, labels = Variable(mat), Variable(mat_DT), Variable(node_features), Variable(node_features_DT), Variable(labels)

val_loss = [float('inf')]
parm = {}

def train(epoch):
    t1 = torch.zeros((1, 82, args.nout))
    t2 = torch.zeros((1, 82, args.nout))
    t = time.time()
    model.train()
    optimizer.zero_grad()

    for i in idx_train:
        tmp = model(node_features[i, :, :], mat[i, :, :]).view(1, 82, args.nout) * 100
        tmp2 = model(node_features_DT[i, :, :], mat[i, :, :]).view(1, 82, args.nout) * 100
        t1 = torch.cat((t1, tmp))
        t2 = torch.cat((t2, tmp2))
    tensorfMR = t1[1:, :, ]
    tensorDMI = t2[1:, :, ]
    print("DMI shape: ", tensorDMI.shape)
    tensorfMR = torch.mean(tensorfMR, 1)  # mean pooling
    tensorDMI = torch.mean(tensorDMI, 1)
    print('DMI: ', tensorDMI, '\n ----- \n')
    print('fMR: ', tensorfMR, '\n ----- \n')
    # tensorfMR = torch.nn.functional.normalize(tensorfMR, p=2, dim=0) # L2 norm
    # tensorDMI = torch.nn.functional.normalize(tensorDMI, p=2, dim=0)
    final_tensor = torch.concat((tensorfMR, tensorDMI), dim=1)  # concat
    mlp = MLP(in_features=tensorDMI.shape[-1] * 2, hid_features=args.hidden, out_features=2)
    final_tensor = mlp(final_tensor)
    # final_tensor = torch.mul(tensorDMI, tensorfMR)  # concat and MlP
    print(tensorDMI.requires_grad)
    acc_train = accuracy(final_tensor, labels[idx_train])
    print('labels: ', labels[idx_train].shape)
    # output1 = torch.mm(tensorfMR, tensorfMR.T) - torch.mm(tensorDMI, tensorDMI.T)
    # output2 = torch.mm(tensorfMR, tensorDMI.T)
    criterion = CloseFormLoss(tensorDMI, tensorfMR)
    param = model.state_dict()['out_att.W'].clone().detach().numpy()
    loss_train = criterion(param, labels[idx_train])
    loss_train.backward()
    optimizer.step()


    # if not args.fastmode:
    #     # Evaluate validation set performance separately,
    #     # deactivates dropout during validation run.
    #     t1 = torch.zeros((1, 2))
    #     t2 = torch.zeros((1, 2))
    #     model.eval()
    #     for i in idx_val:
    #
    #         tmp = model(node_features[i, :, :], mat[i, :, :]).view(1, 2)
    #         tmp2 = model(node_features_DT[i, :, :], mat[i, :, :]).view(1, 2)
    #         t1 = torch.cat((t1, tmp))
    #         t2 = torch.cat((t2, tmp2))
    #     tensorfMR_val = t1[1:, ]
    #     tensorDMI_val = t2[1:, ]
    #     final_tensor_val = torch.mul(tensorDMI_val, tensorfMR_val)
    #
    #     output1_val = torch.mm(tensorfMR_val, tensorfMR_val.T) - torch.mm(tensorDMI_val, tensorDMI_val.T)
    #     loss_val = (torch.norm(output1_val, p=2) + torch.norm(torch.mm(tensorfMR_val, tensorDMI_val.T), p=2) + nll_loss(final_tensor_val, labels[idx_val]))

    t1 = torch.zeros((1, 82, args.nout))
    t2 = torch.zeros((1, 82, args.nout))

    print("\n-------Validation begins---------\n")

    for i in idx_val:
        tmp = model(node_features[i, :, :], mat[i, :, :]).view(1, 82, args.nout)
        tmp2 = model(node_features_DT[i, :, :], mat[i, :, :]).view(1, 82, args.nout)
        t1 = torch.cat((t1, tmp))
        t2 = torch.cat((t2, tmp2))
    tensorfMR_val = t1[1:, :, ]
    tensorDMI_val = t2[1:, :, ]
    tensorfMR_val = torch.mean(tensorfMR_val, 1)
    tensorDMI_val = torch.mean(tensorDMI_val, 1)

    # tensorfMR_val = torch.nn.functional.normalize(tensorfMR_val, p=2, dim=0)
    # tensorDMI_val = torch.nn.functional.normalize(tensorDMI_val, p=2, dim=0)
    final_tensor = torch.concat((tensorfMR_val, tensorDMI_val), dim=1)
    mlp = MLP(in_features=tensorDMI.shape[-1] * 2, hid_features=args.hidden, out_features=2)
    final_tensor_val = mlp(final_tensor)
    # final_tensor_val = torch.mul(tensorDMI_val, tensorfMR_val)
    # output1 = torch.mm(tensorfMR_val, tensorfMR_val.T) - torch.mm(tensorDMI_val, tensorDMI_val.T)
    # output2 = torch.mm(tensorfMR_val, tensorDMI_val.T)
    criterion = CloseFormLoss(tensorDMI_val, tensorfMR_val)
    param = model.state_dict()['out_att.W'].clone().detach().numpy()
    loss_val = criterion(param, labels[idx_val])
    acc_val = accuracy(final_tensor_val, labels[idx_val])
    print("loss in validation set is : ", loss_val, loss_val.data.item())
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train),
          'loss_val: {:.4f}'.format(loss_val),
          'time: {:.4f}s'.format(time.time() - t))
    print('Training acc: {:.4f}'.format(acc_train),
          'Validation acc: {:.4f}'.format(acc_val))

    return loss_val.data.item()

# test function ( has not been modified yet since the training preds are not good )
def compute_test():
    model1_weight_path = './weights_1/147.pth'
    model2_weight_path = './weights_2/147.pth'
    model.load_state_dict(torch.load(model1_weight_path, map_location='cpu'))
    model.eval()
    t1 = torch.zeros((1, 82, args.nout))
    t2 = torch.zeros((1, 82, args.nout))

    for i in idx_test:
        tmp = model(node_features[i, :, :], mat[i, :, :]).view(1, 82, args.nout)
        tmp2 = model(node_features_DT[i, :, :], mat[i, :, :]).view(1, 82, args.nout)
        t1 = torch.cat((t1, tmp))
        t2 = torch.cat((t2, tmp2))
    tensorfMR_val = t1[1:, :, ]
    tensorDMI_val = t2[1:, :, ]
    tensorfMR_val = torch.mean(tensorfMR_val, 1)
    tensorDMI_val = torch.mean(tensorDMI_val, 1)
    tensorfMR = torch.nn.functional.normalize(tensorfMR_val, p=2, dim=0)
    tensorDMI = torch.nn.functional.normalize(tensorDMI_val, p=2, dim=0)
    final_tensor = torch.concat((tensorfMR_val, tensorDMI_val), dim=1)
    mlp = MLP(in_features=tensorDMI.shape[-1] * 2, hid_features=args.hidden, out_features=2)
    final_tensor= mlp(final_tensor)
    loss_test = criterion(tensorfMR, tensorDMI, labels[idx_test])
    acc_test = accuracy(final_tensor, labels[idx_test])
    print(final_tensor, labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.data.item()),
          "accuracy= {:.4f}".format(acc_test.data.item()))


is_train = True
# Train model
if is_train:
    t_total = time.time()
    loss_values = []
    bad_counter = 0
    best = 1e12
    best_epoch = 0
    for epoch in range(args.epochs):
        loss_values.append(train(epoch))

        # torch.save(model.state_dict(), '{}.pkl'.format(epoch))
        if loss_values[-1] < best:
            best = loss_values[-1]
            best_epoch = epoch
            bad_counter = 0
            torch.save(model.state_dict(), "./weights_1/{}.pth".format(epoch))

        else:
            bad_counter += 1
            print('bad_counter for now: ', bad_counter)

        if bad_counter == args.patience:
            break

        files = glob.glob('*.pkl')
        for file in files:
            epoch_nb = int(file.split('.')[0])
            if epoch_nb < best_epoch:
                os.remove(file)
        print('Best so far: ', best)
        print('this epoch: ', loss_values[-1])

    files = glob.glob('*.pkl')
    for file in files:
        epoch_nb = int(file.split('.')[0])
        if epoch_nb > best_epoch:
            os.remove(file)

    print("Optimization Finished!")
    print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

    # Restore best model
    print('Loading {}th epoch'.format(best_epoch))
    model.load_state_dict(torch.load('{}.pkl'.format(best_epoch)))


else:
    compute_test()