import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Flatten
from keras.applications import MobileNetV2
from keras import optimizers
from keras.models import Sequential

data = pd.read_csv('./dataset/processed/age.csv')

data = data.astype({'age' : str, 'path' : str})

data['path'] = data['path'] + '.jpg'

X, y = train_test_split(data, test_size=0.1, random_state=1969)

train_genrator = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.3)


test_genrator = ImageDataGenerator(rescale=1./255,
                                    shear_range=0.2,
                                    zoom_range=0.3)

train_images = train_genrator.flow_from_dataframe(X, 
                                                  directory='./dataset/processed/images/',
                                                  x_col='path',
                                                  y_col='age',
                                                  target_size=(224,224),
                                                  batch_size=32)

#test_images = train_genrator.flow_from_dataframe(y, 
#                                                  directory='./dataset/processed/images/',
#                                                  x_col='path',
#                                                  y_col='age',
#                                                  target_size=(224,224),
#                                                  batch_size=32)

# Making the model
model = Sequential()

# For this model I'm using InceptionResNetV2
# I'll use imagenet weights here also
mobile = MobileNetV2(include_top=False,
                          weights="imagenet", 
                          input_shape=(224,224,3),
                          pooling="max")

# Adding the mobile model and configuting the output layer
model.add(mobile)
model.add(Dense(units=101, activation="softmax"))

# Compiling the model
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

model.fit_generator(train_images,
                    steps_per_epoch=2800,
                    epochs=2)

model.save('weights/age.h5')

