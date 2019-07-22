import cv2
import pandas as pd
import random

wiki = pd.read_csv('./dataset/processed/wiki_meta.csv')
imdb = pd.read_csv('./dataset/processed/imdb_meta.csv')
selfie = pd.read_csv('./dataset/processed/selfie_meta.csv')

wiki = wiki.values
imdb = imdb.values
selfie = selfie.values

data = []
counter = 0

for img in wiki:
    image = cv2.imread('./dataset/unprocessed/' + img[2], 1)
    
    if random.randint(0, 121) == 5:
        print('--Resizing image ' + str(counter))
        
    image = cv2.resize(image, (224,224))
    cv2.imwrite('./dataset/processed/images/' + str(counter) + '.jpg', image)
    data.append([img[0], img[1], counter])
    counter += 1

for img in imdb:
    image = cv2.imread('./dataset/unprocessed/' + img[2], 1)

    if random.randint(0, 121) == 5:
        print('--Resizing image ' + str(counter))
        
    image = cv2.resize(image, (224,224))
    cv2.imwrite('./dataset/processed/images/' + str(counter) + '.jpg', image)
    data.append([img[0], img[1], counter])
    counter += 1
    
for img in selfie:
    image = cv2.imread('./dataset/unprocessed/' + img[1], 1)

    if random.randint(0, 121) == 5:
        print('--Resizing image ' + str(counter))
        
    image = cv2.resize(image, (224,224))
    cv2.imwrite('./dataset/processed/images/' + str(counter) + '.jpg', image)
    data.append([-1, img[0], counter])
    counter += 1
    
df = pd.DataFrame(data)
df.columns = ['age', 'gender', 'path']

# Shuffling the data
df = df.sample(frac=1)

# Storing as csv file
df.to_csv('./dataset/processed/meta.csv', index=False)