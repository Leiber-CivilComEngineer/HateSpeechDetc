import pandas as pd
from sentence_transformers import SentenceTransformer

# 模型路径
model_name_or_path = 'sentence-transformers/distiluse-base-multilingual-cased-v1'

# 中文数据文件路径
Chinese_file = "./data/Chinese/predicted/combined.csv"

# 英文数据文件路径
English_file = "./data/English/CallMeSexist/predicted/combined.csv"

# 输出文件路径
output_file = './data/sentence_emb/Eng_Chi_sentence_emb.csv'

# 加载Sentence-BERT模型
model = SentenceTransformer(model_name_or_path)

# 读取中文数据
Chinese_data = pd.read_csv(Chinese_file)

# 提取中文文本列
Chinese_texts = Chinese_data['text'].tolist()

# 使用模型生成中文文本的嵌入
Chinese_embeddings = model.encode(Chinese_texts)

# 读取英文数据
English_data = pd.read_csv(English_file)

# 提取英文文本列
English_texts = English_data['text'].tolist()

# 使用模型生成英文文本的嵌入
English_embeddings = model.encode(English_texts)

English_data["embedding"] = English_embeddings.tolist()
Chinese_data["embedding"] = Chinese_embeddings.tolist()

res_df = pd.concat([English_data, Chinese_data], ignore_index=True)

# res_df['embedding'] = res_df['embedding'].astype(object)

res_df.to_csv(output_file, index=False)

