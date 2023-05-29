# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import pandas as pd
import numpy as np  
import util
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Word2Vec
from gensim.models.fasttext import FastText
from transformers import BertTokenizer, BertModel
import torch


np.set_printoptions(threshold=np.inf)  


#Chinese emb!!!!!!

# def fit_bow(df_col):
#     vectorizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
#     X = vectorizer.fit_transform(df_col)
#     return X, vectorizer

# def tfidf_emb(df):
#     vectorizer = TfidfVectorizer()
#     documents = df['text'].values
#     tfidf_matrix = vectorizer.fit(documents)
#     df["emb"] = pd.DataFrame(tfidf_matrix)
#     return df

def tfidf_emb(df, vector_size=100):
    vectorizer = TfidfVectorizer()
    documents = df['text'].values
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    X_embedded = []
    for i in range(len(documents)):
        doc_embedding = tfidf_matrix[i].toarray().flatten()
        if len(doc_embedding) < vector_size:
            doc_embedding = np.pad(doc_embedding, (0, vector_size - len(doc_embedding)), 'constant')
        else:
            doc_embedding = doc_embedding[:vector_size]
        X_embedded.append(doc_embedding)
    
    df['emb'] = X_embedded
    return df

def word2vec_averaged_emb(df, vector_size=100, averaged=True, max_token=30):
    documents = [doc.split() for doc in df["text"]]
    model = Word2Vec(documents, vector_size=vector_size, window=5, min_count=5, workers=4)
    X_embedded = []
    for doc in documents:
        doc_embedding = []
        for word in doc:
            if word in model.wv:
                doc_embedding.append(model.wv[word])
        if averaged:
            if doc_embedding:
                doc_embedding = sum(doc_embedding) / len(doc_embedding)
            else:
                doc_embedding = [0] * vector_size
        else:
            doc_len = len(doc_embedding)
            if doc_len > max_token:
                doc_embedding = doc_embedding[:max_token]
            else:
                for i in range(max_token-doc_len):
                    doc_embedding.append([0]*vector_size)
        X_embedded.append(doc_embedding)
    df['emb'] = X_embedded
    return df


def fasttext_emb(df, vector_size=100, averaged=True, max_token=30):
    documents = [doc.split() for doc in df["text"]]
    model = FastText(documents, vector_size=vector_size, window=5, min_count=5, workers=4)
    X_embedded = []
    for doc in documents:
        doc_embedding = []
        for word in doc:
            if word in model.wv:
                doc_embedding.append(model.wv[word])
        if averaged:
            if doc_embedding:
                doc_embedding = sum(doc_embedding) / len(doc_embedding)
            else:
                doc_embedding = [0] * vector_size
        else:
            doc_len = len(doc_embedding)
            if doc_len > max_token:
                doc_embedding = doc_embedding[:max_token]
            else:
                for i in range(max_token-doc_len):
                    doc_embedding.append([0]*vector_size)
        X_embedded.append(doc_embedding)
    df['emb'] = X_embedded
    return df


if __name__ == "__main__":
    import os
    pwd = "/home/leiber/Codings/individual_pro/new/"
    os.chdir(pwd)
    import json
    with open('conf.json') as f:
        data = json.load(f)
    dataset = data['datasets']['SWSR']
    train_test_ratio = data['general']['train_test_ratio']
    root_path = dataset["root_path"]

    root_path = "./data/Chinese/"
    df = pd.read_csv(root_path+"clean.csv")
    # print(df)
    # tfidf_matrix = tfidf_emb(df)
    # X_embedded = word2vec_averaged_emb(df, averaged=False)
    # X_embedded = fasttext_emb(df, averaged=False)
    X_embedded = bert_emb(df)
    print(X_embedded)

 








    # util.split_df(df, train_test_ratio, root_path)
    
    # csv_path = "/home/leiber/Codings/individual_pro/new/data/chinese/SexComment.csv"
    # txt_path = "/home/leiber/Codings/individual_pro/new/data/chinese/chinese_stopword.txt" 
    # stop_words = load_data.load_stop_word(txt_path)
    # df = load_data.load_csv(csv_path)
    # df = load_data.generate_clean_comment(df, load_data.div_and_stop_remove, stop_words)
    # df.to_csv("/home/leiber/Codings/individual_pro/new/data/chinese/clean.csv", index=False)
    # util.split_df(df, 0.7)
    # X, vectorizer = fit_bow(df["text"])
    # tfidf = fit_tfidf(X, vectorizer)
    # tfidf_dense = tfidf.todense()
    # print(tfidf_dense[0])
    


    


