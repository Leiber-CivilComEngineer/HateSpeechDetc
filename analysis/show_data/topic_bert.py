# -*- encoding:utf-8 -*-
#author: Leiber Baoqian Lyu#

import pandas as pd
from bertopic import BERTopic

df = pd.read_csv("./data/sentence_emb/Eng_Chi_sentence_emb.csv")
df = df[df["source"] == "SWSR"]
des_path = "./result/topic_analysis/SWSR_topic.csv"

# df = df[df["source"] == "CallMe"]
# des_path = "./result/topic_analysis/CallMe_topic.csv"

topic_model = BERTopic(language="multilingual", top_n_words=5, calculate_probabilities=False, verbose=True)

embeddings = df['text'].tolist()
topics, _ = topic_model.fit_transform(embeddings)
df['topic'] = topics
unique_topics = df['topic'].unique()

result_df = pd.DataFrame(columns=['topic', 'correct_num', 'wrong_num', 'tp', 'tn', 'fp', 'fn', "accuracy", 'keywords'])

for topic in unique_topics:
    topic_df = df[df['topic'] == topic]  # 获取属于当前主题的数据子集
    total_samples = topic_df.shape[0]

    tp = topic_df[(topic_df['true_label'] == 1) & (topic_df['predict'] == 1)].shape[0]  # 统计正确样本数量
    tn = topic_df[(topic_df['true_label'] == 0) & (topic_df['predict'] == 0)].shape[0]  # 统计正确样本数量
    fp = topic_df[(topic_df['true_label'] == 0) & (topic_df['predict'] == 1)].shape[0]
    fn = topic_df[(topic_df['true_label'] == 1) & (topic_df['predict'] == 0)].shape[0]

    # 获取主题的关键词
    topic_keywords = topic_model.get_topic(topic)

    result_df = result_df.append({
        'topic': topic,
        'correct_num': tp+tn,
        'wrong_num': fp+fn,
        'tp': tp,
        'tn': tn,
        'fp': fp,
        'fn': fn,
        'accuracy': (tp+tn)/total_samples,
        'keywords': topic_keywords
    }, ignore_index=True)

result_df = result_df.sort_values(by='accuracy', ascending=False)
result_df.to_csv(des_path, index=False)

    