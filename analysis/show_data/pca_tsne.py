# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import ast

emb_path = "./data/sentence_emb/Eng_Chi_sentence_emb.csv"
df = pd.read_csv(emb_path)


df['embedding'] = df['embedding'].apply(ast.literal_eval)

features = np.array(df["embedding"].tolist())
# 使用 PCA 进行降维
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# 使用 t-SNE 进行降维
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

df["pca1"] = pca_result[:, 0]
df["pca2"] = pca_result[:, 1]
df["tsne1"] = tsne_result[:, 0]
df["tsne2"] = tsne_result[:, 1]

# # 根据真实标签和预测标签分割数据
correct_ENG = df[(df["true_label"] == df["predict"]) & (df["source"] == "CallMe")]
incorrect_ENG = df[(df["true_label"] != df["predict"]) & (df["source"] == "CallMe")]
correct_CNH = df[(df["true_label"] == df["predict"]) & (df["source"] == "SWSR")]
incorrect_CNH = df[(df["true_label"] != df["predict"]) & (df["source"] == "SWSR")]

# only consider positive
# correct_ENG = df[(df["true_label"] == 1) & (df["predict"] == 1) & (df["source"] == "CallMe")] 
# incorrect_ENG = df[(df["true_label"] == 1) & (df["predict"] == 0) & (df["source"] == "CallMe")]
# correct_CNH = df[(df["true_label"] == 1) & (df["predict"] == 1) & (df["source"] == "SWSR")] 
# incorrect_CNH = df[(df["true_label"] == 1) & (df["predict"] == 0) & (df["source"] == "SWSR")]

# # 绘制四类数据的分布情况
plt.figure(figsize=(20, 16))
plt.scatter(correct_ENG["pca1"], correct_ENG["pca2"], c="cyan", label="correct_ENG")
plt.scatter(incorrect_ENG["pca1"], incorrect_ENG["pca2"], c="blue", label="incorrect_ENG")
plt.scatter(correct_CNH["pca1"], correct_CNH["pca2"], c="red", label="correct_CNH")
plt.scatter(incorrect_CNH["pca1"], incorrect_CNH["pca2"], c="orange", label="incorrect_CNH")
plt.legend()
plt.title("Distribution of Four Categories (PCA)")
plt.xlabel("PCA Dimension 1")
plt.ylabel("PCA Dimension 2")
plt.show()

plt.figure(figsize=(20, 16))
plt.scatter(correct_ENG["tsne1"], correct_ENG["tsne2"], c="cyan", label="correct_ENG")
plt.scatter(incorrect_ENG["tsne1"], incorrect_ENG["tsne2"], c="blue", label="incorrect_ENG")
plt.scatter(correct_CNH["tsne1"], correct_CNH["tsne2"], c="red", label="correct_CNH")
plt.scatter(incorrect_CNH["tsne1"], incorrect_CNH["tsne2"], c="orange", label="incorrect_CNH")

plt.legend()
plt.title("Distribution of Four Categories (t-SNE)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.show()


# 分析模块

# positive_ENG = df[(df["true_label"] == 1) & (df["source"] == "CallMe")]
# positive_ENG.to_csv("temp.csv")
# positive_CHN = df[(df["true_label"] == 1) & (df["source"] == "SWSR")]

# combined_df = pd.DataFrame(columns=["text", "group", "predict"])

# for i, c in enumerate(ENG_clusters):
#     cluster_df = positive_ENG[(positive_ENG["tsne1"] >= c[0]) & (positive_ENG["tsne1"] <= c[1]) & (positive_ENG["tsne2"] >= c[2]) & (positive_ENG["tsne2"] <= c[3])]
#     print(cluster_df)
#     exit()
#     cluster_df["group"] = i
#     cluster_df = cluster_df[["text", "group", "predict"]]
#     combined_df = pd.concat([combined_df, cluster_df], ignore_index=True)

# for i, c in enumerate(CHN_clusters):
#     cluster_df = positive_CHN[(positive_CHN["tsne1"] >= c[0]) & (positive_CHN["tsne1"] <= c[1]) & (positive_CHN["tsne2"] >= c[2]) & (positive_CHN["tsne2"] <= c[3])]
#     cluster_df["group"] = i
#     cluster_df = cluster_df[["text", "group", "predict"]]
#     combined_df = pd.concat([combined_df, cluster_df], ignore_index=True)

# combined_df.to_csv("./result/positive_grouped.csv")
