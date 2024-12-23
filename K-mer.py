import pandas as pd
from itertools import product
# 设置k值
k = 3
# 读取CSV文件
df = pd.read_csv('MDI-2/data_orig/miRNA-jianji.csv', names=['miRNA_Name', 'Search_Result'], skiprows=1)
# 初始化一个空的DataFrame来存储第一个结果
results = pd.DataFrame(columns=['miRNA_Name', 'k-mer_vector'])
# 生成所有可能的k-mer
all_kmers = [''.join(bases) for bases in product('ACGU', repeat=k)]
# 为每个k-mer分配一个索引
kmer_to_index = {kmer: index for index, kmer in enumerate(all_kmers)}
# 遍历每一行
for index, row in df.iterrows():
    mirna_name = row['miRNA_Name']
    sequence = row['Search_Result']

    # 初始化一个长度为64的零向量
    kmer_vector = [0] * len(all_kmers)

    # 计算序列长度
    seq_length = len(sequence)

    # 遍历序列，更新k-mer向量
    for i in range(seq_length - k + 1):
        kmer = sequence[i:i + k]
        if kmer in kmer_to_index:
            kmer_vector[kmer_to_index[kmer]] += 1

            # 归一化k-mer向量
    if seq_length - k + 1 > 0:
        kmer_vector = [freq / (seq_length - k + 1) for freq in kmer_vector]

        # 将结果作为新行添加到DataFrame中
    temp_df = pd.DataFrame([[mirna_name, kmer_vector]], columns=['miRNA_Name', 'k-mer_vector'])
    results = pd.concat([results, temp_df], ignore_index=True)

# 将结果保存到新的CSV文件
results.to_csv('k-mer_vectors.csv', index=False)
