# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import pandas as pd
from sentence_transformers import SentenceTransformer, models
import torch

# 可调节的参数
model_name_or_path = 'sentence-transformers/distiluse-base-multilingual-cased-v1'
Chinese_file = "./data/Chinese/predicted/combined.csv"
English_file = "./data/English/CallMeSexist/predicted/combined.csv"
output_file = './data/sentence_emb/Eng_Chi_sentence_emb.csv'
embedding_dim = 128  # 嵌入向量的维度
use_gpu = True  # 是否使用GPU加速

# 设置使用的设备
device = torch.device('cuda' if use_gpu and torch.cuda.is_available() else 'cpu')

# 加载模型
word_embedding_model = models.Transformer(model_name_or_path)
pooling_model = models.Pooling(word_embedding_dimension=embedding_dim)
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
model.to(device)


df_Eng = pd.read_csv(English_file)
df_Chi = pd.read_csv(Chinese_file)
dfs = [df_Eng, df_Chi]

for df in dfs:
    # 提取文本列
    texts = df['text'].tolist()

    # 对文本进行嵌入
    embeddings = model.encode(texts, device=device, convert_to_tensor=True)

    # 将嵌入结果添加到DataFrame中
    df['embedding'] = [embedding.cpu().tolist() for embedding in embeddings]
    print(len(df['embedding'].loc[1]))
    exit()

df = pd.concat([df_Eng, df_Chi])
# print(df.columns)
# 保存结果到CSV文件
df.to_csv(output_file, index=False)
