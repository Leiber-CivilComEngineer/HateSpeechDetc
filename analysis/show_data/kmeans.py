# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import ast

emb_path = "./data/sentence_emb/Eng_Chi_sentence_emb.csv"
df = pd.read_csv(emb_path)

df['embedding'] = df['embedding'].apply(ast.literal_eval)

features = np.array(df["embedding"].tolist())

n_clusters = 4  # 设定聚类的簇数
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(features)

df["cluster"] = clusters

fig, axs = plt.subplots(nrows=n_clusters, ncols=1, figsize=(8, 6*n_clusters))

for i in range(n_clusters):
    cluster_data = df[df["cluster"] == i]
    class_counts = cluster_data.groupby(["true_label", "predict"]).size().unstack(fill_value=0)
    
    class_counts.plot(kind="bar", stacked=True, ax=axs[i])
    axs[i].set_title("Cluster {}".format(i))
    axs[i].set_xlabel("True Label")
    axs[i].set_ylabel("Count")
    axs[i].legend(["Predicted Negative", "Predicted Positive"])

plt.tight_layout()
plt.show()
