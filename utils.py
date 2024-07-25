import csv
import numpy as np
import torch


from sklearn.preprocessing import scale

import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def read_csv(path):
    with open(path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)
        matrix = np.array(data).astype(np.int64)
        return matrix



def multiomics_data():
    # miRNA_fea
    kmer = np.genfromtxt("data_orig/k-mer_vectors.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    miRNA = kmer[:, 0]
    miRNA_number = len(miRNA)
    kmer = np.array(kmer)
    kmer = scale(np.array(kmer[:, 1:], dtype=float))
    # pca = PCA(n_components=64)
    # kmer = pca.fit_transform(kmer)
    kmer_adj = sim_graph(kmer, miRNA_number)


    fusion_miRNA_fea = kmer
    fusion_miRNA_adj = kmer_adj

    # drug_fea

    PC = np.genfromtxt("data_orig/fea_PubChem.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    drug = PC[:, 0]
    drug_number = len(drug)
    PC = np.array(PC)
    PC = scale(np.array(PC[:, 1:], dtype=float))
    pca = PCA(n_components=32)
    PC = pca.fit_transform(PC)
    PC_adj = sim_graph(PC, drug_number)


    MACCS = np.genfromtxt("data_orig/fea_MACCS.csv", delimiter=',', skip_header=1, dtype=np.dtype(str))
    drug = MACCS[:, 0]
    drug_number = len(drug)
    MACCS = np.array(MACCS)
    MACCS = scale(np.array(MACCS[:, 1:], dtype=float))
    pca = PCA(n_components=32)
    MACCS = pca.fit_transform(MACCS)
    MACCS_adj = sim_graph(MACCS, drug_number)


    fusion_drug_fea = np.concatenate((PC, MACCS), axis=1)
    fusion_drug_adj = np.logical_or(PC_adj, MACCS_adj).astype(int)
    #
    # fusion_drug_fea = MACCS
    # fusion_drug_adj = MACCS_adj

    #加载label
    labellist = []
    # with open('data-human/03label666.txt', 'r') as file:
    with open('data_orig/output_filter-2.txt', 'r') as file:
        lines = file.readlines()
    for line in lines:
        line = line.strip()  # 移除行首和行尾的空白字符
        elements = line.split("\t")  # 使用空格分隔元素
        processed_elements = [int(elements[1]), int(elements[0]), int(elements[2])]  # 互换位置
        labellist.append(processed_elements)
    labellist = torch.Tensor(labellist)
    print("drug miRNA lable:", labellist.shape)

    miRNA_feat, miRNA_adj = torch.FloatTensor(fusion_miRNA_fea), torch.FloatTensor(fusion_miRNA_adj)
    drug_feat, drug_adj = torch.FloatTensor(fusion_drug_fea), torch.FloatTensor(fusion_drug_adj)
    return miRNA_feat, miRNA_adj, drug_feat, drug_adj, labellist

def sim_graph(omics_data, miRNA_number):
    sim_matrix = np.zeros((miRNA_number, miRNA_number), dtype=float)
    adj_matrix = np.zeros((miRNA_number, miRNA_number), dtype=float)

    for i in range(miRNA_number):
        for j in range(i + 1):
            sim_matrix[i, j] = np.dot(omics_data[i], omics_data[j]) / (
                        np.linalg.norm(omics_data[i]) * np.linalg.norm(omics_data[j]))
            sim_matrix[j, i] = sim_matrix[i, j]

    for i in range(miRNA_number):
        topindex = np.argsort(sim_matrix[i])[-10:]
        for j in topindex:
            adj_matrix[i, j] = 1
    return adj_matrix

