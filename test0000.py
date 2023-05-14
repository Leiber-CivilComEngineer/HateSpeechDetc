# import pandas as pd

# # 读取 CSV 文件
# df = pd.read_csv('/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/call_me_sexist.csv')

# # 剔除不符合逗号分隔条件的行和空行
# df = df.dropna()  # 剔除空行
# df = df[df['column_name'].str.count(',') == 6]  # 剔除不符合逗号分隔条件的行

# # 保存剔除后的结果为新的 CSV 文件
# df.to_csv('/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/sexist_test.csv', index=False)

import pandas as pd

# def remove_invalid_rows(ori_path, tar_path):
#     df = pd.read_csv(ori_path)
#     header = df.columns.tolist()
#     df = df.dropna(subset=header)
#     df.to_csv(tar_path, index=False)

# # 使用示例
# remove_invalid_rows('/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/call_me_sexist.csv', "/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/sexist_test.csv")
ori_path = '/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/call_me_sexist.csv'
df = pd.read_csv(ori_path)
# df.to_csv("/home/leiber/Codings/individual_pro/new/data/English/CallMeSexist/sexist_test02.csv", index=False)
pd.set_option('display.max_colwidth', 300)
print(df[12:13]['text'])