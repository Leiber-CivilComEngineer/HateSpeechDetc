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

def tfidf_emb(df):
    vectorizer = TfidfVectorizer()
    documents = df['text'].values
    tfidf_matrix = vectorizer.fit(documents)
    df["emb"] = pd.DataFrame(tfidf_matrix)
    return df

def word2vec_averaged_emb(df, vector_size=100, averaged=True):
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
        X_embedded.append(doc_embedding)
    df['emb'] = X_embedded
    return df


def fasttext_emb(df, vector_size=100, averaged=True):
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
        X_embedded.append(doc_embedding)
    df['emb'] = X_embedded
    return df
    
def bert_emb(df):
    #not function
    model_name = 'bert-base-chinese'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)
    embeddings = []
    for document in df['text']:
        # 使用tokenizer将文档分词并添加特殊标记
        tokens = tokenizer.tokenize(document)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        
        # 将分词转换为词汇索引
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        
        # 将词汇索引转换为PyTorch张量
        input_ids = torch.tensor(input_ids).unsqueeze(0)  # 添加批处理维度
        
        # 计算BERT模型的词嵌入
        with torch.no_grad():
            outputs = model(input_ids)
            # 获取最后一层的隐藏状态作为词嵌入
            embeddings.append(outputs[0][:, 0, :].squeeze().tolist())  # 取CLS的隐藏状态作为整个文档的表示

    # 将词嵌入添加到DataFrame中
    embeddings_df = pd.DataFrame(embeddings)
    return embeddings_df


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
    


    


