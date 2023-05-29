# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import numpy as np
import pandas as pd
import imbedding
import util
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, Embedding
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score

def rnn_model(train_df, develop_df, test_df):
    # 训练集
    train_embeddings = np.array(train_df['emb'].tolist())
    train_labels = np.array(train_df['label'].tolist())

    # 验证集
    dev_embeddings = np.array(develop_df['emb'].tolist())
    dev_labels = np.array(develop_df['label'].tolist())

    # 测试集
    test_embeddings = np.array(test_df['emb'].tolist())
    test_labels = np.array(test_df['label'].tolist())

    # 将每个文本的词向量填充到相同长度
    max_length = 30  # 假设每个文本最大长度为30
    train_embeddings = pad_sequences(train_embeddings, maxlen=max_length)
    dev_embeddings = pad_sequences(dev_embeddings, maxlen=max_length)
    test_embeddings = pad_sequences(test_embeddings, maxlen=max_length)
    # 模型定义
    input_shape = train_embeddings.shape[1:]

    model = Sequential()
    model.add(SimpleRNN(128, input_shape=input_shape))  # input_shape=(max_length, embedding_dim)
    model.add(Dense(1, activation='sigmoid'))

    # 模型编译
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 训练模型
    model.fit(train_embeddings, train_labels, validation_data=(dev_embeddings, dev_labels), epochs=10, batch_size=32)

    # 评估模型
    loss, accuracy = model.evaluate(test_embeddings, test_labels, batch_size=32)
    print('Test loss:', loss)
    print('Test accuracy:', accuracy)

    # 预测测试集的标签
    test_predictions = model.predict(test_embeddings)
    test_predictions = np.round(test_predictions).flatten()

    # 计算精确度、召回率和 F1 分数
    precision = precision_score(test_labels, test_predictions)
    recall = recall_score(test_labels, test_predictions)
    f1 = f1_score(test_labels, test_predictions)

    print('Precision:', precision)
    print('Recall:', recall)
    print('F1-score:', f1)

    return model

if __name__ == "__main__":
    import os
    pwd = "/home/leiber/Codings/individual_pro/new/"
    os.chdir(pwd)
    import json
    with open('conf.json') as f:
        data = json.load(f)
    dataset = data['datasets']['CallMeSexist']
    # dataset = data['datasets']['SWSR']
    train_test_ratio = data['general']['train_test_ratio']
    root_path = dataset["root_path"]
    df = pd.read_csv(root_path+"clean.csv")
    df = imbedding.word2vec_averaged_emb(df, vector_size=100, averaged=False)
    train_df, develop_df, test_df= util.split_df(df=df, train_ratio=0.6, develop_ratio=0.2, test_ratio=0.2, random_state=1)
    rnn_model(train_df, develop_df, test_df)