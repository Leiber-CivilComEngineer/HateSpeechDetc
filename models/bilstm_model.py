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
from keras.layers import LSTM, Bidirectional, Dense
from keras_preprocessing.sequence import pad_sequences
from sklearn.metrics import precision_score, recall_score, f1_score

def bilstm_model(train_df, develop_df, test_df):
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
    input_shape = train_embeddings.shape[1:]
    model = Sequential()
    model.add(Bidirectional(LSTM(units=256, return_sequences=True), input_shape=input_shape))
    model.add(Bidirectional(LSTM(units=128)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # 打印模型概述
    model.summary()
    model.fit(train_embeddings, train_labels, validation_data=(dev_embeddings, dev_labels), epochs=5, batch_size=32)

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
    train_test_ratio = data['general']['train_test_ratio']
    root_path = dataset["root_path"]
    df = pd.read_csv(root_path+"clean.csv")


    df = imbedding.word2vec_averaged_emb(df, vector_size=100, averaged=False)
    train_df, develop_df, test_df= util.split_df(df=df, train_ratio=0.6, develop_ratio=0.2, test_ratio=0.2, random_state=1)
    bilstm_model(train_df, develop_df, test_df)