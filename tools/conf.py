# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#
#Ref: https://github.com/karlhl/Bert-classification-pytorch/blob/main/tools/config.py#


class Config(object):
    def __init__(self):
        self.config_dict = {
            "data_path": {
                "trainingSet_path": "/home/leiber/Codings/individual_pro/new/data/chinese/train_data.csv",
                # "valSet_path": "./data/sentiment/sentiment.valid.data",
                "testingSet_path": "/home/leiber/Codings/individual_pro/new/data/chinese/test_data.csv"
            },

            "BERT_path": {
                "file_path": 'hfl/chinese-bert-wwm',
                "config_path": 'hfl/chinese-bert-wwm',
                "vocab_path": 'hfl/chinese-bert-wwm',
            },

            "training_rule": {
                "max_length": 300, # 输入序列长度，别超过512
                "hidden_dropout_prob": 0.3,
                "num_labels": 2, # 几分类个数
                "learning_rate": 1e-5,
                "weight_decay": 1e-2,
                "batch_size": 64
            },

            "result": {
                "model_save_path": './result/',
                "config_save_path": './result/',
                "vocab_save_path": './result/'
            }
        }

    def get(self, section, name):
        return self.config_dict[section][name]