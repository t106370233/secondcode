#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 23 14:17:39 2019

@author: wuchi
"""

import os,sys
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model, load_model
from keras.layers import *
from keras.callbacks import *
from keras.utils import np_utils
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
import h5py
import pandas as pd
from keras.callbacks import EarlyStopping, History
#%%
images=[]
labels= []
name= []
#%%
def read_images_labels(path,i):
    for file in os.listdir(path):
        abs_path = os.path.abspath(os.path.join(path, file))
        if os.path.isdir(abs_path):
            i+=1
            temp = os.path.split(abs_path)[-1]
            name.append(temp)
            read_images_labels(abs_path,i)
            amount = int(len(os.listdir(path)))
            sys.stdout.write('\r'+'>'*(i)+' '*(amount-i)+'[%s%%]'%(i*100/amount)+temp)
 
 
        if file.endswith('.jpg'):
            image=cv2.resize(cv2.imread(abs_path),(64,64))
            images.append(image)
            labels.append(i-1)
    return images, labels , name
#%%

def read_main(path):
    images, labels ,name = read_images_labels(path,i=0)
    images = np.array(images,dtype=np.float32)/255
    labels = np_utils.to_categorical(labels, num_classes=20)
    np.savetxt('name.txt', name, delimiter = ' ',fmt="%s")
    return images, labels
#labels=3，經過np_utils.to_categorical，會轉換為labels= [0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

images, labels=read_main('train/characters-20')
#%%
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.1)
#%%
print('\r')
print("-----------------------")
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)
print("-----------------------")
#%%
epochs= 100
batch_size= 200
#%%
model = Sequential()

model.add(Conv2D(64, kernel_size=3 , padding='same',activation='relu', input_shape=X_train.shape[1:]))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(128, kernel_size=3, padding='same',activation='relu'))
model.add(Conv2D(128, kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(256, kernel_size=3, padding='same',activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same',activation='relu'))
model.add(Conv2D(256, kernel_size=3, padding='same',activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))



model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.5))
#model.add(Dense(256,activation='relu'))
#model.add(Dropout(0.5))
model.add(Dense(20,activation='softmax'))
model.summary()
#%%
sgd = SGD(lr=0.01, momentum=0.9, decay=1e-6, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])

datagen = ImageDataGenerator(zoom_range=0.1,width_shift_range=0.05,height_shift_range=0.05,horizontal_flip=True)
datagen.fit(X_train)
#%%
file_name=str(epochs)+'_'+str(batch_size)
#%%
early_stopping = EarlyStopping(monitor='val_accuracy', patience= 6, verbose=2)
#%%
h = model.fit_generator(datagen.flow(X_train, y_train, batch_size=batch_size),
                    steps_per_epoch=epochs, epochs=epochs,
                    validation_data = (X_test, y_test ), verbose = 1, callbacks=[early_stopping])
#%%
model.save(file_name+'.h5')
score = model.evaluate(X_test, y_test, verbose=1)
print(score)
#%%

def read_images(path):
    images=[]
    for i in range(990):
        image=cv2.resize(cv2.imread(path+str(i+1)+'.jpg'),(64,64))
        images.append(image)

    images=np.array(images,dtype=np.float32)/255
    return images

def transform(listdir,label,lenSIZE):
    label_str=[]
    for i in range (lenSIZE):
        temp=listdir[label[i]]
        label_str.append(temp)

    return label_str

images = read_images('test/test/')
model = load_model('100_250.h5')
#%%
predict = model.predict_classes(images, verbose=1)
print(predict)
label_str=transform(np.loadtxt('name.txt',dtype='str'),predict,images.shape[0])

df = pd.DataFrame({"character": label_str})
df.index = np.arange(1, len(df) + 1)
df.index.names = ['id']
df.to_csv('test.csv')
