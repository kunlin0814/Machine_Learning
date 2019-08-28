#!/usr/bin/env python
#Step1 Importing and building the required model
##import_dependencies
import pandas as pd
import numpy as np
import os
import keras
import matplotlib.pyplot as plt
from keras.layers import Dense,GlobalAveragePooling2D
from keras.applications import vgg16
from keras.preprocessing import image
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

##build_model
base_model=vgg16.VGG16(weights='imagenet',include_top=False) #imports the vgg16 model and discards the last 1000 neuron layer.
x=base_model.output
###fine-tune
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(4,activation='softmax')(x) #final layer with softmax activation


##make_model
model=Model(inputs=base_model.input,outputs=preds)

##model_layers
for i,layer in enumerate(model.layers):
	print(i,layer.name)

###or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
	layer.trainable=False
for layer in model.layers[20:]:
	layer.trainable=True

#Step2 loading the training data into the ImageDataGenerator
##img_dat_gen
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
img_width, img_height = 224, 224
train_generator=train_datagen.flow_from_directory('/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/4Cluster/model/Train/',
                                                 target_size=(img_width, img_height),
                                                 color_mode='rgb',
                                                 batch_size=100,
                                                 class_mode='categorical',
                                                 shuffle=True)
print (train_generator.class_indices)
#Step3 training the model on the dataset
##train_step
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=64)

model.save('/scratch/kh31516/TCGA/Stomach/TCGA-HE-Image/4Cluster/model/my_model_All4Train_different_epoch_batch.h5')
