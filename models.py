# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import imbedding
import load_data 
import util
import pandas as pd
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb




def svm_model(x_train, y_train, x_test, y_test):
    model = SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    return model

def decision_tree_model(x_train, y_train, x_test, y_test):
    model = DecisionTreeClassifier(random_state=1)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    return model

def random_forest_model(x_train, y_train, x_test, y_test):
    model = RandomForestClassifier(random_state=1)
    model.fit(x_train, y_train)
    print(model.score(x_test, y_test))
    return model

def xgb_model(x_train, y_train, x_test, y_test):
    model = xgb.XGBClassifier(max_depth=4, learning_rate=0.1, n_estimators=100, silent=False, objective='binary:logistic')
    model.fit(x_train, y_train)
    y_predict = model.predict(x_test)
    confusion_mat = metrics.confusion_matrix(y_test, y_predict)
    print('accuracy:', metrics.accuracy_score(y_test, y_predict))
    print("confusion_matrix is: ", confusion_mat)
    print('clasification report:', metrics.classification_report(y_test, y_predict))
    return model



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
    # util.split_df(df, train_test_ratio, root_path)
    df = imbedding.word2vec_averaged_emb(df, vector_size=100, averaged=True)
    df_train, df_test = util.split_df(df=df, train_ratio=0.8, test_ratio=0.2, random_state=1)
    # df_train = pd.read_csv(root_path+"train.csv")
    # df_test = pd.read_csv(root_path+"train.csv")


    # root_path = "./data/Chinese/"
    # df = pd.read_csv(root_path+"clean.csv")
    # train_test_ratio = 0.7
    # util.split_df(df, train_test_ratio, root_path)

    # train_data = pd.read_csv("/home/leiber/Codings/individual_pro/new/data/chinese/test_data.csv")
    # test_data = pd.read_csv("/home/leiber/Codings/individual_pro/new/data/chinese/train_data.csv")
    # total_data = pd.read_csv("/home/leiber/Codings/individual_pro/new/data/chinese/clean.csv")
    # X_total, vectorizer = imbedding.fit_bow(total_data["clean"])
    # tfidf_transformer = imbedding.fit_tfidf(X_total)


    # bow_train_x = vectorizer.transform(train_data["clean"])
    # bow_test_x = vectorizer.transform(test_data["clean"])
    

    # tfidf_train_x = tfidf_transformer.transform(bow_train_x)
    # tfidf_test_x = tfidf_transformer.transform(bow_test_x)

    # train_y = train_data["label"]
    # test_y = test_data["label"]

    # xgb_model(bow_train_x, train_y, bow_test_x, test_y)
    # print("svm on bow:")
    # svm_model(bow_train_x, train_y, bow_test_x, test_y)
    # print("svm on tfidf:")
    # svm_model(tfidf_train_x, train_y, tfidf_test_x, test_y)
    # print("single tree on tfidf:")
    # decision_tree_model(tfidf_train_x, train_y, tfidf_test_x, test_y)
    # print("RF on tfidf:")
    # random_forest_model(tfidf_train_x, train_y, tfidf_test_x, test_y)