# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import ast

emb_path = "./data/sentence_emb/Eng_Chi_sentence_emb.csv"
df = pd.read_csv(emb_path)

embeddings = np.array(df["embedding"].apply(ast.literal_eval))
vector_dimensions = set(len(vector) for vector in embeddings)
print(vector_dimensions)
print(embeddings[0])
print(len(embeddings[0]))
# print(type(embeddings[0]))
# print(type(embeddings))
exit()


df["combined_label"] = "Real: " + df["true_label"].astype(str) + "correctly predicted: " + df["predict"].astype(str)

tsne = TSNE(n_components=2, random_state=42, perplexity=50)
reduced_data = tsne.fit_transform(embeddings)

unique_labels = df["combined_label"].unique()

colors = ['red', 'green', 'blue', 'purple']

for i, label in enumerate(unique_labels):
    indices = df["combined_label"] == label
    plt.scatter(reduced_data[indices, 0], reduced_data[indices, 1], color=colors[i], label=label)

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")
plt.title("t-SNE Visualization")
plt.legend()
plt.show()