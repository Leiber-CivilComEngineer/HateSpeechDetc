import pandas as pd

train_paths = "/home/leiber/Codings/individual_pro/new/data/English/ConvAbuse/ConvAbuseEMNLPtrain.csv"
test_paths = "/home/leiber/Codings/individual_pro/new/data/English/ConvAbuse/ConvAbuseEMNLPtest.csv"
vaild_paths = "/home/leiber/Codings/individual_pro/new/data/English/ConvAbuse/ConvAbuseEMNLPvalid.csv"

paths = [train_paths, vaild_paths, test_paths]
combined_data = pd.DataFrame()

for file_path in paths:
    df = pd.read_csv(file_path)
    combined_data = pd.concat([combined_data, df])

combined_data['text'] = combined_data['prev_agent'] + ' ' + combined_data['prev_user'] + ' ' + combined_data['agent'] + ' ' + combined_data['user']


combined_data.to_csv('/home/leiber/Codings/individual_pro/new/data/English/ConvAbuse/ConvAbuse.csv', index=False)