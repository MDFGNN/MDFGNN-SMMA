import copy
import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from torch.autograd import Variable
from sklearn.metrics import r2_score
from scipy.stats import pearsonr

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit

from utils import multiomics_data
from model import MultiDeep
import torch.utils.data as Dataset

from datetime import datetime


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--epochs', type=int, default=120, help='Number of epochs to train.')
parser.add_argument('--lr', type=float, default=0.0001, help='Initial learning rate.')
parser.add_argument('--batch', type=int, default=128, help='Number of batch size')
parser.add_argument('--hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--nb_heads', type=int, default=2, help='Number of head attentions.')
parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
parser.add_argument('--patience', type=int, default=5, help='Patience')
parser.add_argument('--dropout', type=float, default=0.5, help='dropout')

args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.manual_seed(args.seed)
    used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
    cached_memory = torch.cuda.memory_reserved()  # 缓存的GPU内存量
    print(f"GPU success，服务器GPU已分配：{used_memory / 1024 ** 3:.2f} GB，已缓存：{cached_memory / 1024 ** 3:.2f} GB")
else:
    device = torch.device("cpu")

# Load data
miRNA_features, miRNA_adj, drug_features, drug_adj, sample_set = multiomics_data()
miRNA_features, miRNA_adj = Variable(miRNA_features), Variable(miRNA_adj)
drug_features, drug_adj = Variable(drug_features), Variable(drug_adj)
miRNA_features, miRNA_adj = miRNA_features.to(device), miRNA_adj.to(device)
drug_features, drug_adj = drug_features.to(device), drug_adj.to(device)

used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
cached_memory = torch.cuda.memory_reserved()   #缓存的GPU内存量
print(f"数据上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")

# Model and optimizer
model = MultiDeep(nmiRNA=miRNA_features.shape[0],
                  ndrug=drug_features.shape[0],
                  nmiRNAfeat=miRNA_features.shape[1],
                  ndrugfeat=drug_features.shape[1],
                  nhid=args.hidden,
                  nheads=args.nb_heads,
                  alpha=args.alpha,
                  dropout=args.dropout)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)

# loss_func = nn.MSELoss()
loss_func =nn.BCEWithLogitsLoss()
loss_func.to(device)
used_memory = torch.cuda.memory_allocated()  # 已使用的GPU内存量
cached_memory = torch.cuda.memory_reserved()   #缓存的GPU内存量
print(f"loss函数上传成功，服务器GPU已分配：{used_memory / 1024**3:.2f} GB，已缓存：{cached_memory / 1024**3:.2f} GB")
best_value = [0, 0, 1]
model_date = datetime.now().strftime("%Y-%m-%d %H-%M-%S")


def train(epoch, index_tra, y_tra, index_val, y_val):
    time_begin = time.time()

    output_train = [model_date]
    output_valid = [model_date]

    tra_dataset = Dataset.TensorDataset(index_tra, y_tra)
    train_dataset = Dataset.DataLoader(tra_dataset, batch_size=args.batch, shuffle=True)

    model.train()
    for index_trian, y_train in train_dataset:
        y_train = y_train.to(device)
        output = model(miRNA_features, miRNA_adj, drug_features, drug_adj, index_trian.numpy().astype(int), device)
        loss_train = loss_func(output, y_train)
        y_tpred = torch.sigmoid(output)
        optimizer.zero_grad()
        loss_train.backward()
        optimizer.step()

    model.eval()
    loss_valid, RMSE_valid, PCC_valid, R2_valid = [], [], [], []
    val_dataset = Dataset.TensorDataset(index_val, y_val)
    valid_dataset = Dataset.DataLoader(val_dataset, batch_size=args.batch, shuffle=True)
    pred_valid, true_valid = [], []
    for index_valid, y_valid in valid_dataset:
        y_valid = y_valid.to(device)
        output = model(miRNA_features, miRNA_adj, drug_features, drug_adj, index_valid.numpy().astype(int), device)
        loss = loss_func(output, y_valid)
        y_vpred =torch.sigmoid(output)
        pred_valid.extend(y_vpred.cpu().detach().numpy())
        true_valid.extend(y_valid.cpu().detach().numpy())

    time_over = time.time()

    loss_valid = mean_squared_error(true_valid, pred_valid)
    RMSE_valid = np.sqrt(loss_valid)
    MAE_valid = mean_absolute_error(true_valid, pred_valid)
    PCC_valid = pearsonr(true_valid, pred_valid)[0]
    R2_valid = r2_score(true_valid, pred_valid)
    pred_valid_binary = [1 if x >= 0.5 else 0 for x in pred_valid]
    Precision_valid = precision_score(true_valid, pred_valid_binary)
    AUC_valid = roc_auc_score(true_valid ,pred_valid)
    AUPR_valid = average_precision_score(true_valid,pred_valid)
    output_valid.append(epoch+1)
    output_valid.append(loss_valid)
    output_valid.append(RMSE_valid)
    output_valid.append(MAE_valid)
    output_valid.append(PCC_valid)
    output_valid.append(R2_valid)
    output_valid.append(AUC_valid)
    output_valid.append(AUPR_valid)
    output_valid.append(Precision_valid)


    pred_train = y_tpred.cpu().detach().numpy()
    true_train = y_train.cpu().detach().numpy()
    RMSE_train = np.sqrt(loss_train.item(), out=None)
    MAE_train = mean_absolute_error(true_train, pred_train)
    PCC_train = pearsonr(true_train, pred_train)[0]
    R2_train = r2_score(true_train, pred_train)

    output_train.append(epoch+1)
    output_train.append(loss_train.item())
    output_train.append(RMSE_train)
    output_train.append(MAE_train)
    output_train.append(PCC_train)
    output_train.append(R2_train)


    if (epoch+1)  % 10 == 0:
        print('Epoch: {:04d}'.format(epoch + 1),
              '\n loss_train: {:.4f}'.format(loss_train.item()),
              'RMSE_train: {:.4f}'.format(RMSE_train),
              'MAE_train: {:.4f}'.format(MAE_train),
              'PCC_train: {:.4f}'.format(PCC_train),
              'R2_train: {:.4f}'.format(R2_train),


              '\n loss_valid: {:.4f}'.format(loss_valid),
              'RMSE_valid: {:.4f}'.format(RMSE_valid),
              'MAE_valid: {:.4f}'.format(MAE_valid),
              'PCC_valid: {:.4f}'.format(PCC_valid),
              'R2_valid: {:.4f}'.format(R2_valid),
              'Precision_valid:{:.4f}'.format(Precision_valid),
              'AUC_valid: {:.4f}'.format(AUC_valid),
              'AUPR_valid: {:.4f}\n'.format(AUPR_valid ))


    return  AUC_valid,AUPR_valid


def compute_test(index_test, y_test):
    model.eval()
    loss_test, PCC_test, RMSE_test, R2_test = [], [], [], []
    pred_test, true_test = [], []
    dataset = Dataset.TensorDataset(index_test, y_test)
    test_dataset = Dataset.DataLoader(dataset, batch_size=args.batch, shuffle=False)  # Set shuffle to False
    test_results = []  # Create an empty list to store all test results
    for index_test, y_test in test_dataset:
        y_test = y_test.to(device)
        output = model(miRNA_features, miRNA_adj, drug_features, drug_adj, index_test.numpy().astype(int), device)
        loss_test = loss_func(output, y_test)
        y_pred = torch.sigmoid(output)
        pred_test.extend(y_pred.cpu().detach().numpy())
        true_test.extend(y_test.cpu().detach().numpy())

        test_results.extend(list(zip(index_test.numpy(), y_test.cpu().detach().numpy(),
                                     y_pred.cpu().detach().numpy())))  # Add this batch's results to the list

    # # Now test_results contains the results for the entire test set
    # for idx, true_val, pred_val in test_results:
    #     print(f"Index: {idx}, True Value: {true_val},    Predicted Value: {pred_val}")
    #
    # # Save the test results to a CSV file
    # test_results_df = pd.DataFrame(test_results, columns=['Index', 'True Value', 'Predicted Value'])
    # test_results_df.to_csv('test_results.csv', index=False)

    loss_test = mean_squared_error(true_test, pred_test)
    RMSE_test = np.sqrt(loss_test)
    MAE_test = mean_absolute_error(true_test, pred_test)
    PCC_test = pearsonr(true_test, pred_test)[0]
    R2_test = r2_score(true_test, pred_test)
    AUC_test = roc_auc_score(true_test,pred_test)
    AUPR_test = average_precision_score(true_test,pred_test)
    pred_test_binary = [1 if x >= 0.5 else 0 for x in pred_test]
    Precision_test = precision_score(true_test, pred_test_binary)
    Recall_test = recall_score(true_test, pred_test_binary)
    F1_score_test = f1_score(true_test, pred_test_binary)

    with open("MultiMDI", 'a') as f:
        f.write(str(model_date) + " " + str(RMSE_test) + " " + str(MAE_test) + " " + str(PCC_test) + " " + str(R2_test)
                + " " + str(AUC_test) + " " + str(AUPR_test)
                + " " + str(Precision_test) + " " + str(Recall_test) + " " + str(F1_score_test) + "\n")

    print("Test set results:",
          "\n loss_test: {:.4f}".format(loss_test),
          "RMSE_test: {:.4f}".format(RMSE_test),
          'MAE_test: {:.4f}'.format(MAE_test),
          "PCC_test: {:.4f}".format(PCC_test),
          "R2_test: {:.4f}".format(R2_test),
          "AUC_test: {:.4f}".format(AUC_test),
          "AUPR_test: {:.4f}".format(AUPR_test),
          "Precision_test: {:.4f}".format(Precision_test),
          "Recall_test: {:.4f}".format(Recall_test),
          "F1_score_test: {:.4f}".format(F1_score_test)
          )


# Train model
time_begin = time.time()


sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
labels = sample_set[:, 2]
# 使用StratifiedShuffleSplit对象来划分数据
for train_index, test_index in sss.split(np.arange(sample_set.shape[0]), labels):
    train_set, test_set = sample_set[train_index], sample_set[test_index]

# Prepare the test set
index_test, y_test = test_set[:, :2], test_set[:, 2]
y_test = Variable(y_test, requires_grad=True)

# Initialize the model
model.to(device)

# Prepare the KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# Prepare the features and labels
X = train_set[:, :2]
y = train_set[:, 2]
# Initialize the best AUC and best model state
best_auc = 0
best_aupr = 0
best_model_state = None

# Perform 10-fold cross validation
for fold, (train_index, valid_index) in enumerate(skf.split(X, y)):
    print("Starting fold {}"
          "-----------------------------------------".format(fold + 1))

    # Prepare the training set and validation set for this fold
    index_train, y_train = X[train_index], y[train_index]
    index_valid, y_valid = X[valid_index], y[valid_index]
    y_train, y_valid = Variable(y_train, requires_grad=True), Variable(y_valid, requires_grad=True)

    # Train the model on this fold
    auc_valid = [0]
    aupr_valid = [0]
    # Precision_valid = [0]
    bad_counter = 0
    fold_best_auc = 0
    fold_best_aupr = 0
    fold_best_model_state = None
    for epoch in range(args.epochs):
        avg_auc_valid,avg_aupr_valid= train(epoch, index_train, y_train, index_valid, y_valid)
        auc_valid.append(avg_auc_valid)
        aupr_valid.append(avg_aupr_valid)
        # Precision_valid.append(avg_Precision_valid)
        if abs(auc_valid[-1] - auc_valid[-2] ) < 0.0005 and abs(aupr_valid[-1] - aupr_valid[-2]) < 0.0005:
            bad_counter += 1
        else:
            bad_counter = 0
        if bad_counter >= args.patience:
            break
        # If this epoch's performance is better than the best so far, save the model state
        if auc_valid[-1] > fold_best_auc and aupr_valid[-1] > fold_best_aupr:
            fold_best_auc = auc_valid[-1]
            fold_best_aupr = aupr_valid[-1]
            fold_best_model_state = copy.deepcopy(model.state_dict())
    print("Training stopped at epoch", epoch + 1)
    print("Finished fold {}".format(fold + 1))

    # If this fold's performance is better than the best so far, save the model state
    if fold_best_auc > best_auc and fold_best_aupr > best_aupr:
        best_auc = fold_best_auc
        best_aupr = fold_best_aupr
        best_model_state = fold_best_model_state

# Load the best model state
model.load_state_dict(best_model_state)

# Testing
compute_test(index_test, y_test)

time_total = time.time() - time_begin
print("Total time: {:.4f}s".format(time_total))
