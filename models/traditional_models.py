# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import imbedding
import load_data 
import util
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import xgboost as xgb



def lr_model(x_train, y_train, x_test, y_test):
    model = LogisticRegression()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----lr-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model

def svm_model(x_train, y_train, x_test, y_test):
    model = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----SVM-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model

def decision_tree_model(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----decision_tree-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model

def random_forest_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----random_forest-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model

def gbdt_model(x_train, y_train, x_test, y_test):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----GBDT-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model

def xgb_model(x_train, y_train, x_test, y_test):
    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, silent=False, objective='binary:logistic')
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----XGB-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return model

def naive_bayes_model(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----naive_bayes-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)

    return model

def adaboost_model(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    accuracy = model.score(x_test, y_test)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    print("-----adaboost-----")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1-Score:", f1)
    return model



if __name__ == "__main__":
    # import os
    # pwd = "/home/leiber/Codings/individual_pro/new/"
    # os.chdir(pwd)
    # import json
    # with open('conf.json') as f:
    #     data = json.load(f)
    # dataset = data['datasets']['SWSR']
    # train_test_ratio = data['general']['train_test_ratio']
    # root_path = dataset["root_path"]
    # df = pd.read_csv(root_path+"clean.csv")
    # # df = imbedding.word2vec_averaged_emb(df, vector_size=100, averaged=True)
    # df = imbedding.fasttext_emb(df, vector_size=100, averaged=True)
    # # df = imbedding.tfidf_emb(df)
    # df_train, df_test = util.split_df(df=df, train_ratio=0.8, develop_ratio=0, test_ratio=0.2, random_state=1)
    # x_train = df_train['emb']
    # y_train = df_train['label']
    # x_test = df_test['emb']
    # y_test = df_test['label']
    # x_train = np.array(x_train.values.tolist())
    # y_train = np.array(y_train.values.tolist())
    # x_test = np.array(x_test.values.tolist())
    # y_test = np.array(y_test.values.tolist())
    # lr_model(x_train, y_train, x_test, y_test)
    # svm_model(x_train, y_train, x_test, y_test)
    # decision_tree_model(x_train, y_train, x_test, y_test)
    # random_forest_model(x_train, y_train, x_test, y_test)
    # xgb_model(x_train, y_train, x_test, y_test)
    # gbdt_model(x_train, y_train, x_test, y_test)
    # naive_bayes_model(x_train, y_train, x_test, y_test)
    # adaboost_model(x_train, y_train, x_test, y_test)
    
    
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
    # df = imbedding.word2vec_averaged_emb(df, vector_size=100, averaged=True)
    df = imbedding.fasttext_emb(df, vector_size=100, averaged=True)
    # df = imbedding.tfidf_emb(df)
    df_train, df_test = util.split_df(df=df, train_ratio=0.8, develop_ratio=0, test_ratio=0.2, random_state=1)
    x_train = df_train['emb']
    y_train = df_train['label']
    x_test = df_test['emb']
    y_test = df_test['label']
    x_train = np.array(x_train.values.tolist())
    y_train = np.array(y_train.values.tolist())
    x_test = np.array(x_test.values.tolist())
    y_test = np.array(y_test.values.tolist())
    lr_model(x_train, y_train, x_test, y_test)
    svm_model(x_train, y_train, x_test, y_test)
    decision_tree_model(x_train, y_train, x_test, y_test)
    random_forest_model(x_train, y_train, x_test, y_test)
    xgb_model(x_train, y_train, x_test, y_test)
    gbdt_model(x_train, y_train, x_test, y_test)
    naive_bayes_model(x_train, y_train, x_test, y_test)
    adaboost_model(x_train, y_train, x_test, y_test)
