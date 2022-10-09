import pickle
import random
import numpy as np
from glob import glob
from PIL import Image
from itertools import permutations
from keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import keras 
from keras.datasets import mnist 
from keras.models import Sequential 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from keras import backend as K 
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
# Import helper functions from utils.py
import utils
import os
import json

training_data_folder = '../train'
corrected_folder = '../unscrambled/all_correct'

class ImageBorders:
    def __init__(self, image_array):
        self.top_border = image_array[0]
        self.bottom_border = image_array[-1]
        self.left_border = image_array[:,0]
        self.right_border = image_array[:,-1]



def read_data_and_make_pickle ():

    # make lists that have enough size upfront to avoid append
    # all_images = [None for i in range(10000)]
    # scrambled_set = [1 for i in range(10000)]
    all_images = []
    scrambled_set = []
    total_data = pd.DataFrame(columns= ['image', 'correct'])
    
    for folder_name in os.listdir(training_data_folder):

        for img_name in os.listdir(os.path.join(training_data_folder, folder_name)):
            if folder_name != "0123":
                img_path = os.path.join(training_data_folder, folder_name, img_name)
                correct_sequence = folder_name
                
                
                img = load_img(f'{img_path}', target_size=(128, 128))
                img_array = img_to_array(img)
                all_images.append(img_array)
                scrambled_set.append(1)


    for img_name in os.listdir(corrected_folder):
        img_path = os.path.join(corrected_folder, img_name)

        img = load_img(f'{img_path}', target_size=(128, 128))
        img_array = img_to_array(img)
        all_images.append(img_array)

        scrambled_set.append(0)

    print("finishned readifg the stuff!!!!1")

    with open("complete_data.pickle", "wb") as out_file:
        pickle.dump({
            "all_images": all_images,
            "scrambled_set": scrambled_set
        }, out_file)

    print("outputted the stuff!!!")



#read_data_and_make_pickle()
print("DONE PICKLINF!!!!!!")

#######################################################3   THA PROGRAM START HERE

# Read pickle
# print("\n\nreading tha pickel...")
# complete_data = None
# with open("complete_data.pickle", "rb") as in_file:
#     complete_data = pickle.load(in_file)

# print("finshed resding the pickLEE!!!\n")


# print(complete_data.keys())
# print(type(complete_data["all_images"]))
# X_train, X_test, y_train, y_test = train_test_split(np.asarray(complete_data["all_images"]).astype('float32'), np.asarray(complete_data["scrambled_set"]).astype('float32'), test_size=0.9)

TRAIN_COUNT = 100
correct = os.listdir("../unscrambled/all_correct")
random.shuffle(correct)
incorrect = os.listdir("../scrambled/yep")
random.shuffle(incorrect)
filenames = [""] * (2 * TRAIN_COUNT)
categories = ['correct'] * TRAIN_COUNT + ['incorrect'] * TRAIN_COUNT
for index, i in enumerate(correct[:TRAIN_COUNT]):
    # print('/mnt/c/Users/naviy/Desktop/aaaaaaa/puzzle_solverrrr/unscrambled/all_correct/' + i)
    filenames[index] = '/mnt/c/Users/naviy/Desktop/aaaaaaa/puzzle_solverrrr/unscrambled/all_correct/' + i

for index, i in enumerate(incorrect[:TRAIN_COUNT]):
    filenames[index+TRAIN_COUNT] = '/mnt/c/Users/naviy/Desktop/aaaaaaa/puzzle_solverrrr/scrambled/yep/' + i



# print(filenames[500])
# print("i should have printed something")
# print(filenames[1500])
# print(categories[250])
# print(categories[1250])
df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

df_quadrants = pd.DataFrame(columns=['top_left', 'top_right', 'bottom_left', 'bottom_right', 'category'])

for index, row in df[['filename', 'category']].iterrows():
    img = load_img(row[0], target_size=(128, 128))
    quadrants = utils.get_uniform_rectangular_split(img_to_array(img), 2, 2)
    df_quadrants.loc[len(df_quadrants.index)] = [quadrants[0], quadrants[1], quadrants[2], quadrants[3], row[1]]

# X = np.array(df_quadrants[['top_left', 'top_right', 'bottom_left', 'bottom_right']])
# y = np.array(df_quadrants['category'])
# X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=0.2, random_state=0)
# print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


# from sklearn.preprocessing import MinMaxScaler
# sc=MinMaxScaler()
# X_train_scaled = sc.fit_transform(X_train)
# X_test_scaled = sc.transform(X_test)
# X_train_scaled.shape


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D,\
     Dropout,Flatten,Dense,Activation,\
     BatchNormalization
model=Sequential()
model.add(Conv2D(32,(3,3),activation='relu',input_shape=(64, 64, 4)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(64,(3,3),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))
# model.add(Conv2D(128,(3,3),activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2,2)))
# model.add(Dropout(0.5))
# model.add(Flatten())
# model.add(Dense(512,activation='relu'))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
model.add(Dense(2,activation='softmax'))
#softmax big
#softmax probability
#sigmoid

#model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
  optimizer='rmsprop',metrics=['accuracy'])

print(model.summary())


from keras.callbacks import EarlyStopping, ReduceLROnPlateau
earlystop = EarlyStopping(patience = 10)
learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 2,verbose = 1,factor = 0.5,min_lr = 0.0000001)
callbacks = [earlystop,learning_rate_reduction]



train_df,validate_df = train_test_split(df_quadrants,test_size=0.20,random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)
total_train=train_df.shape[0]
total_validate=validate_df.shape[0]
batch_size=30


train_datagen = ImageDataGenerator()
train_generator = train_datagen.flow_from_dataframe(train_df, x_col=['top_left', 'top_right', 'bottom_left', 'bottom_right'],y_col='category',
                                                 target_size = (64, 64, 4),
                                                 class_mode='binary',
                                                 batch_size=batch_size)
validation_datagen = ImageDataGenerator()
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    x_col='filename',
    y_col='category',
    target_size=(64, 64,4),
    class_mode='categorical',
    batch_size=batch_size
)
test_datagen = ImageDataGenerator()
test_filenames = os.listdir("/mnt/c/Users/naviy/Desktop/aaaaaaa/puzzle_solverrrr/src/example_images")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]
test_generator = train_datagen.flow_from_dataframe(train_df,x_col=[['top_left', 'top_right', 'bottom_left', 'bottom_right']],y_col='category',
                                                 target_size=(64, 64,4),
                                                 class_mode='categorical',
                                                 batch_size=batch_size)

epochs=10

history = model.fit(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)

model.save("submissionshuffle.h5")

# img_width, img_height = 128, 128
# img = image.load_img('/mnt/c/Users/naviy/Desktop/aaaaaaa/puzzle_solverrrr/unscrambled/all_correct/1.png', target_size = (128, 128))
# img = image.img_to_array(img)
# img = np.expand_dims(img, axis = 0)

# prediction = model.predict(img)
# print(prediction[0][0],  prediction[0][1])
# print(np.argmax(prediction, axis=-1))
