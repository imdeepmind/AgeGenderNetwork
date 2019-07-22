import pandas as pd

data = pd.read_csv('./dataset/processed/meta.csv', error_bad_lines=False)

data = data.drop(['gender'], axis=1)

data = data[data['age'] != -1]

data.to_csv('./dataset/processed/age.csv', index=False)