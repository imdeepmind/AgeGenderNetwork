## IMPORTING THE DEPENDENCIES
import pandas as pd
import numpy as np

selfie_data = './dataset/unprocessed/Selfie-dataset/selfie_dataset.txt'
file = open(selfie_data, 'r')
selfie_file = file.read()

selfie_file_lines = selfie_file.split('\n')

un_selfie_data = []
for selfie in selfie_file_lines:
    temp = selfie.split(' ')
    if len(temp) > 3:
        un_selfie_data.append(['Male' if temp[3] == '0' else 'Female', 'Selfie-dataset/images/' + temp[0] + '.jpg']) 

selfie = pd.DataFrame(un_selfie_data)
selfie.columns = ['gender', 'path']


# Shuffling the data
selfie = selfie.sample(frac=1)

# Storing as csv file
selfie.to_csv('./dataset/processed/selfie_meta.csv', index=False)