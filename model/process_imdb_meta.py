## IMPORTING THE DEPENDENCIES
import pandas as pd
import numpy as np
from scipy.io import loadmat
from utils import compare_date, clean_full_date

## LOADING THE .MAT FILE
imdb_mat = './dataset/unprocessed/imdb_crop/imdb.mat'
imdb_data = loadmat(imdb_mat)
imdb = imdb_data['imdb']

## PROCESS THE DATA

# The data stores the dob as matlab serial number in the .mat file. Along with that it stores
# the dob in the filename of each image also. In the image filename, it is stored as normal date. 
# So reading that is simpler

up_full_path = imdb[0][0][2][0] # Path of the images
up_gender = imdb[0][0][3][0] # Gender
up_face_score1 = imdb[0][0][6][0] # Score of the first image
up_face_score2 = imdb[0][0][7][0] # Score of the second image (NaN if there is no second image)

# Getting the gender
p_gender = []

for gender in up_gender:
    if gender == 1:
        p_gender.append('Male')
    elif gender == 0:
        p_gender.append('Female')
    else:
        p_gender.append('Unknown')
        
# Getting the dob and path
p_dob = []
p_path = []

for path in up_full_path:
    temp = path[0].split('_')
    photo_taken = temp[3].split('.')[0]
    dob = clean_full_date(temp[2])
    p_dob.append(compare_date(dob, photo_taken))
    p_path.append(path[0])


# Stacking the data
imdb_processed_data = np.vstack((p_dob, p_gender, p_path, up_face_score1, up_face_score2)).T

## SAVING THE DATA

# Making a datagrame from the data
cols = ['age', 'gender', 'path', 'f1_score', 'f2_score']
imdb_df = pd.DataFrame(imdb_processed_data)
imdb_df.columns = cols

# Filtering the data
imdb_df = imdb_df[imdb_df['f1_score'] != '-inf']
imdb_df = imdb_df[imdb_df['f2_score'] == 'nan']

# Removing the f1 and f2 score
imdb_df = imdb_df.drop(['f1_score', 'f2_score'], axis=1)

# Shuffling the data
imdb_df = imdb_df.sample(frac=1)

# Storing as csv file
imdb_df.to_csv('./dataset/processed/imdb_meta.csv', index=False)
