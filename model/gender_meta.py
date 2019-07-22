import pandas as pd

data = pd.read_csv('./dataset/processed/meta.csv', error_bad_lines=False)

data = data.drop(['age'], axis=1)

data = data[data['gender'] != 'Unknown']

data.to_csv('./dataset/processed/gender.csv', index=False)