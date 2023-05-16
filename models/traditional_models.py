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
    accuracy = model.score(x_test, y_test)
    print("-----lr-----")
    print("Accuracy:", accuracy)
    return model

def svm_model(x_train, y_train, x_test, y_test):
    model = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    model.fit(x_train, y_train)
    print("---SVM---")
    print(model.score(x_test, y_test))
    return model

def decision_tree_model(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    print("---decision_tree---")
    print(model.score(x_test, y_test))
    return model

def random_forest_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    print("---random_forest---")
    print(model.score(x_test, y_test))
    return model

def gbdt_model(x_train, y_train, x_test, y_test):
    model = GradientBoostingClassifier()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print("-----gbdt-----")
    print("Accuracy:", accuracy)
    return model

def xgb_model(x_train, y_train, x_test, y_test):
    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, silent=False, objective='binary:logistic')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    print("-----xgb-----")
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    # print("confusion_matrix is: ", confusion_mat)
    # print('clasification report:', metrics.classification_report(y_test, y_predict))
    return model

def naive_bayes_model(x_train, y_train, x_test, y_test):
    model = GaussianNB()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print("-----naive_bayes-----")
    print("Accuracy:", accuracy)
    return model

def adaboost_model(x_train, y_train, x_test, y_test):
    model = AdaBoostClassifier()
    model.fit(x_train, y_train)
    accuracy = model.score(x_test, y_test)
    print("-----adaboost-----")
    print("Accuracy:", accuracy)
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
    # df_train, df_test = util.split_df(df=df, train_ratio=0.8, develop_ratio=0, test_ratio=0.2, random_state=1)
    # x_train = df_train['emb']
    # y_train = df_train['label']
    # x_test = df_test['emb']
    # y_test = df_test['label']
    # x_train = np.array(x_train.values.tolist())
    # y_train = np.array(y_train.values.tolist())
    # x_test = np.array(x_test.values.tolist())
    # y_test = np.array(y_test.values.tolist())
    # svm_model(x_train, y_train, x_test, y_test)
    # decision_tree_model(x_train, y_train, x_test, y_test)
    # random_forest_model(x_train, y_train, x_test, y_test)
    # xgb_model(x_train, y_train, x_test, y_test)
    # lr_model(x_train, y_train, x_test, y_test)
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
    df_train, df_test = util.split_df(df=df, train_ratio=0.8, develop_ratio=0, test_ratio=0.2, random_state=1)
    x_train = df_train['emb']
    y_train = df_train['label']
    x_test = df_test['emb']
    y_test = df_test['label']
    x_train = np.array(x_train.values.tolist())
    y_train = np.array(y_train.values.tolist())
    x_test = np.array(x_test.values.tolist())
    y_test = np.array(y_test.values.tolist())
    svm_model(x_train, y_train, x_test, y_test)
    decision_tree_model(x_train, y_train, x_test, y_test)
    random_forest_model(x_train, y_train, x_test, y_test)
    xgb_model(x_train, y_train, x_test, y_test)
    lr_model(x_train, y_train, x_test, y_test)
    gbdt_model(x_train, y_train, x_test, y_test)
    naive_bayes_model(x_train, y_train, x_test, y_test)
    adaboost_model(x_train, y_train, x_test, y_test)
