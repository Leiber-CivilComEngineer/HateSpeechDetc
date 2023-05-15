# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import load_data


import time
import torch
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split



def bert_model(origin_df, language):

    # 设置随机种子，以便结果可重现
    torch.manual_seed(42)

    # 加载预训练的BERT模型和tokenizer
    if language == "Chinese":
        model_name = 'bert-base-chinese'
    elif language == "English":
        model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_df, test_df = train_test_split(origin_df, test_size=0.2, random_state=42)

    # 对训练集文本进行标记化和编码
    train_encoded_inputs = tokenizer(list(train_df['text']), padding=True, truncation=True, max_length=128, return_tensors='pt')
    train_input_ids = train_encoded_inputs['input_ids']
    train_attention_mask = train_encoded_inputs['attention_mask']
    train_labels = torch.tensor(list(train_df['label']))

    # 对测试集文本进行标记化和编码
    test_encoded_inputs = tokenizer(list(test_df['text']), padding=True, truncation=True, max_length=128, return_tensors='pt')
    test_input_ids = test_encoded_inputs['input_ids']
    test_attention_mask = test_encoded_inputs['attention_mask']
    test_labels = torch.tensor(list(test_df['label']))

    # 创建训练集和测试集的数据集和数据加载器
    train_dataset = TensorDataset(train_input_ids, train_attention_mask, train_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_mask, test_labels)
    batch_size = 32
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    # 定义优化器和损失函数
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # loss_fn = torch.nn.CrossEntropyLoss()

    # 训练模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    num_epochs = 5
    epoch_loss = []
    for epoch in range(num_epochs):
        total_loss = 0
        # for batch in train_dataloader:
        for batch_idx, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_mask, labels = batch
            
            optimizer.zero_grad()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}]')  
            # print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}') 
        
        avg_loss = total_loss / len(train_dataloader)
        epoch_loss.append(avg_loss)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}')

    # 评估模型
    model.eval()

    num_correct = 0
    num_samples = 0

    with torch.no_grad():
        for test_batch in test_dataloader:
            test_batch = tuple(t.to(device) for t in test_batch)
            test_input_ids, test_attention_mask, test_labels = test_batch
            
            test_outputs = model(input_ids=test_input_ids, attention_mask=test_attention_mask)
            test_logits = test_outputs.logits
            
            _, predicted_labels = test_logits.max(1)
            num_correct += (predicted_labels == test_labels).sum().item()
            num_samples += test_labels.size(0)

    accuracy = num_correct / num_samples
    print("-------done------")
    for i, loss in enumerate(epoch_loss):
        print("epoch: {}, loss: {}".format(i+1, loss))
    print(f'Test Accuracy: {accuracy:.4f}')


if __name__ == "__main__":
    import os
    pwd = "/home/leiber/Codings/individual_pro/new/"
    os.chdir(pwd)  
    import json
    with open('conf.json') as f:
        data = json.load(f)

    # dataset = data['datasets']['SWSR']
    # root_path = dataset["root_path"]
    # file_name = dataset["file_name"]
    # language = dataset["language"]
    # text_col_name = dataset["text_col_name"]
    # label_col_name = dataset["label_col_name"]
    # true_label = dataset["true_label"]
    # df = load_data.load_original_csv(root_path=root_path,file_name=file_name,text_col_name=text_col_name,label_col_name=label_col_name)
    # start_time = time.time()
    # bert_model(df, language=language)
    # end_time = time.time()
    # execution_time = end_time - start_time
    # print("time used: {:.2f}秒".format(execution_time))

    dataset = data['datasets']['CallMeSexist']
    root_path = dataset["root_path"]
    file_name = dataset["file_name"]
    language = dataset["language"]
    text_col_name = dataset["text_col_name"]
    label_col_name = dataset["label_col_name"]
    true_label = dataset["true_label"]
    df = load_data.load_original_csv(root_path=root_path,file_name=file_name,text_col_name=text_col_name,label_col_name=label_col_name, true_label=true_label)
    start_time = time.time()
    bert_model(df, language=language)
    end_time = time.time()
    execution_time = end_time - start_time
    print("time used: {:.2f}秒".format(execution_time))