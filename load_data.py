# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#
#email: u7351101@anu.edu.au#

import pandas as pd
import numpy as np
import re
import jieba
import spacy
from spacy.lang.en.stop_words import STOP_WORDS


def load_original_csv(root_path, file_name, text_col_name, label_col_name, true_label=None):
    """
        Load the original csv file, only keep text column and label column, transfer label as "0"/"1"
        Save processed df to clean.csv


    Args:
        root_path (string): folder path of original file path, a relative path to the "run.py" file
        file_name (string): original file name
        language (string): Either English dataset or Chinese dataset
        text_col_name (string): Column name for text
        label_col_name (string): Column name for label
        true_label (list): All label that consider to be True, this is only used when multiple label are considered true, Defaults to None.

    Returns:
        df (Dataframe): clean Dataframe containing only text and label
    """

    df = pd.read_csv(root_path+file_name, encoding='utf-8', error_bad_lines=False, skip_blank_lines=True)
    df = df[[text_col_name, label_col_name]]
    if true_label:
        df[label_col_name] = df[label_col_name].apply(lambda x: 1 if x in true_label else 0)    #transfer label to 0/1
    # data["comment_text"] = jieba.cut(data["comment_text"])
    df.rename(columns={text_col_name: 'text'}, inplace=True)
    df.rename(columns={label_col_name: 'label'}, inplace=True)
    return df

def tokenize_remove_stopword(df, root_path, file_name, language, stop_word_path=None):
    """
        tokenize the text and remove stop word
        Save the clean.csv

    Args:
        df (Dataframe): The df that only contain text column and label column
        root_path (string): folder path of original file path, a relative path to the "run.py" file, such as "./data/English/ConvAbuse/"
        file_name (string): original file name
        language (string): Either English dataset or Chinese dataset
        text_col_name (string): Column name for text
        stop_word_path (string): Path of the txt file of stop words, each word takes a line

    Returns:
        df (Dataframe): clean Dataframe containing only tokenized text and label
    """ 

    def line_process_Chinese(text, stop_word):
        """
            A helper function used in df.apply()
            which process one single line of df

        Args:
            text (string): One line in dataframe
            stop_word (list): The list of loaded stop words

        Returns:
            (string): The string of words after tokenization and stop word removal, separated by space
        """
        text = text.replace(' ', '')    #get rid of space
        content_seg = jieba.cut(text)
        pattern = r'[\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）：]+'
        result = [word for word in content_seg if not word in stop_word]
        result = [re.sub(pattern, '', word) for word in result if word]
        return " ".join(result)
    
    def line_process_English(text, stop_word):
        """
            A helper function used in df.apply()
            which process one single line of df

        Args:
            text (string): One line in dataframe
            stop_word (list): The list of loaded stop words

        Returns:
            (string): The string of words after tokenization and stop word removal, separated by space
        """
        doc = nlp(text)
        res = [token.text for token in doc if (not token.is_stop and not re.match(r'^\W+$', token.text) and token.is_alpha)]
        return " ".join(res)
    
    if language == "Chinese":
        stop_word = load_stop_word(stop_word_path) 
        df["text"] = df["text"].apply(line_process_Chinese, args=(stop_word,))
    elif language == "English":
        stop_word = STOP_WORDS
        nlp = spacy.load("en_core_web_sm")
        df["text"] = df["text"].apply(line_process_English, args=(stop_word,))
    else:
        raise ValueError("Language configuration for {} is neither English nor Chinese!".format(file_name)) #catch error
    
    df.replace('', np.nan, inplace=True)
    df = df.dropna()
    df.to_csv("{}clean.csv".format(root_path))  #save clean.csv

    return df

def load_stop_word(stop_word_path, header=None):
    """
        Load stop word file of txt
        each word take one line

    Args:
        stop_word_path (string): Path of the txt file of stop words, each word takes a line
        header (boolean, optional): whether contain header. Defaults to None.

    Returns:
        stop_words (list): list of stop words
    """
    stop_words = pd.read_csv(stop_word_path, sep='/n', header=None)
    stop_words = stop_words[0].values.tolist()
    return stop_words

if __name__ == "__main__": 
    import os
    pwd = "/home/leiber/Codings/individual_pro/new/"
    os.chdir(pwd)  
    import json
    with open('conf.json') as f:
        data = json.load(f)
    dataset = data['datasets']['SWSR']
    stop_word_path = data['general']['Chinese_stop_word']

    root_path = dataset["root_path"]
    file_name = dataset["file_name"]
    language = dataset["language"]
    text_col_name = dataset["text_col_name"]
    label_col_name = dataset["label_col_name"]
    true_label = dataset["true_label"]
    df = load_original_csv(root_path, file_name, text_col_name, label_col_name, true_label=None)
    df = tokenize_remove_stopword(df, root_path, file_name, language, stop_word_path=stop_word_path)


    
    # dataset = data['datasets']['CallMeSexist']
    # stop_word_path = data['general']['English_stop_word']
    # root_path = dataset["root_path"]
    # file_name = dataset["file_name"]
    # language = dataset["language"]
    # text_col_name = dataset["text_col_name"]
    # label_col_name = dataset["label_col_name"]
    # true_label = dataset["true_label"]
    # df = load_original_csv(root_path, file_name, text_col_name, label_col_name, true_label=None)
    # df = tokenize_remove_stopword(df, root_path, file_name, language, stop_word_path=stop_word_path)