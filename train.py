
# coding: utf-8

# ## Import Libraries

# In[1]:

import numpy as np
from keras.backend import set_image_data_format
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.convolutional import Conv1D, Conv2D, MaxPooling2D
from keras.utils.np_utils import to_categorical
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint


# In[2]:

set_image_data_format("channels_first")


# ## Create Data Generator

# In[3]:

batch_size = 32

# input image dimensions
img_rows, img_cols = 100, 100

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        './../Dataset/train_data',
        target_size=(img_rows, img_cols),
        batch_size=batch_size)

validation_generator = test_datagen.flow_from_directory(
        './../Dataset/val_data',
        target_size=(img_rows, img_cols),
        batch_size=batch_size)


# ## Create Training Model

# In[4]:

#number of epochs
nb_epoch = 5
# size of pooling area for max pooling
nb_pool = 2
# convolution kernel size
nb_conv = 3


# In[5]:

MODEL_FILENAME="./../Output/model.h5"

def load_train_model(force=False):
    if not force and os.path.exists(MODEL_FILENAME):
        model=load_model(MODEL_FILENAME)
    else:
        print("Force loading...")
        model = Sequential()
        
        model.add(Conv2D(32, kernel_size=nb_conv,
                                padding='same',
                                input_shape=(3, img_rows, img_cols)))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        
        model.add(Conv2D(64, kernel_size=nb_conv,padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        
        model.add(Conv2D(128, kernel_size=nb_conv,padding='same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
        
        model.add(Dropout(0.25))

        model.add(Flatten())
        
        model.add(Dense(4096))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(1024))
        model.add(Activation('relu'))
        model.add(Dropout(0.5))
        
        model.add(Dense(14))
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    print("Model loaded.")
    return model


# In[ ]:

model=load_train_model(force=True)
model.summary()


# ## Train Model

# In[ ]:

checkpoint=ModelCheckpoint(MODEL_FILENAME, monitor='val_acc', verbose=0, save_best_only=False, mode='auto', period=1)
history=model.fit_generator(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=nb_epoch,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint])

