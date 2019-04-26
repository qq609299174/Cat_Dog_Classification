# Loading packages..
print("Loading packages..\n")
import sys
import pandas as pd
import numpy as np
import random
import os
from keras.preprocessing.image import ImageDataGenerator, load_img
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('TkAgg')  # or whatever other backend that you want
import matplotlib.pyplot as plt
print(os.listdir("CatDog"))

# Read training data
filenames = os.listdir("CatDog/train")
categories = []
for filename in filenames:
	category = filename.split('.')[0]
	if category == 'dog':
		categories.append(1)
	else:
		categories.append(0)

df = pd.DataFrame({'filename':filenames,'category':categories})



print("\nHead of training set \n",df.head())
print("\nTail of training set \n",df.tail())


print("\nTotal in count for each category\n",df['category'].value_counts())

# Initialization variables
FAST_RUN = False
IMAGE_WIDTH = 128
IMAGE_HEIGHT = 128
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 3



# Creating model sequential...
print("\nCreating model sequential...\n")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, Activation, BatchNormalization

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# My local GPU is too slow to run those layers
# model.add(Conv2D(128, (3, 3), activation='relu'))
# model.add(BatchNormalization())
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

# Define your model, loss and optimizer
model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.summary()

# Check-Point
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

earlystop = EarlyStopping(patience=10)

learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc', 
                                            patience=2, 
                                            verbose=1, 
                                            factor=0.5, 
                                            min_lr=0.001)

callbacks = [earlystop, learning_rate_reduction]



# Split training and validation dataset
train_df, validate_df = train_test_split(df, test_size=0.20, random_state=42)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)


print("\nTraining set after spliting\n",train_df['category'].value_counts())
print("\nValidation set after spliting\n",validate_df['category'].value_counts())


total_train = train_df.shape[0]
total_validate = validate_df.shape[0]
batch_size=15


# Image data generator for both training and validation dataset
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1
)


train_df['category']= train_df['category'].apply(str)
train_generator = train_datagen.flow_from_dataframe(
    train_df, 
    "CatDog/train",    # file location
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)

validation_datagen = ImageDataGenerator(rescale=1./255)



validate_df['category']= validate_df['category'].apply(str)
validation_generator = validation_datagen.flow_from_dataframe(
    validate_df, 
    "CatDog/train/",     # file location
    x_col='filename',
    y_col='category',
    target_size=IMAGE_SIZE,
    class_mode='binary',
    batch_size=batch_size
)


example_df = train_df.sample(n=1).reset_index(drop=True)
example_generator = train_datagen.flow_from_dataframe(
	example_df,
	"CatDog/train",     # file location
	x_col='filename',
	y_col='category',
	target_size=IMAGE_SIZE,
	calss_mode='binary')


## Show how image data generator works


print("\nShow how image data generator works...\n")
plt.figure(figsize=(12, 12))
for i in range(0, 15):
    plt.subplot(5, 3, i+1)
    for X_batch, Y_batch in example_generator:
        image = X_batch[0]
        plt.imshow(image)
        break
plt.tight_layout()
plt.show()


epochs=2 if FAST_RUN else 10
history = model.fit_generator(
    train_generator, 
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=total_validate//batch_size,
    steps_per_epoch=total_train//batch_size,
    callbacks=callbacks
)


# Save the model

model.save_weights("model.h5")

#########################################################################
# After tuned model I can use it for prediction on test data. 
# The same process: create testing generator then make the prediction

# predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/batch_size))

# threshold = 0.5
# test_df['probability'] = predict
# test_df['category'] = np.where(test_df['probability'] > threshold, 1,0)
#########################################################################
