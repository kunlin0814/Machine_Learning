
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

##build_model
base_model=vgg16.VGG16(weights='imagenet',include_top=False) #imports the vgg16 model and discards the last 1000 neuron layer.
x=base_model.output
###fine-tune
x=GlobalAveragePooling2D()(x)
x=Dense(1024,activation='relu')(x) #we add dense layers so that the model can learn more complex functions and classify for better results.
x=Dense(1024,activation='relu')(x) #dense layer 2
x=Dense(512,activation='relu')(x) #dense layer 3
preds=Dense(2,activation='softmax')(x) #final layer with softmax activation
###last layer fit 2 group
base_model=vgg16.VGG16(weights=None, include_top=True)
#Add a layer where input is the output of the  second last layer 
x = Dense(2, activation='softmax', name='predictions')(base_model.layers[-2].output)
#Then create the corresponding model 
model = Model(input=base_model.input, output=x)
model.summary()



##make_model
model=Model(inputs=base_model.input,outputs=preds)

##model_layers
for i,layer in enumerate(model.layers):
	print(i,layer.name)

##non_trainable
for layer in model.layers:
	layer.trainable=False
###or if we want to set the first 20 layers of the network to be non-trainable
for layer in model.layers[:20]:
	layer.trainable=False
for layer in model.layers[20:]:
	layer.trainable=True

#Step2 loading the training data into the ImageDataGenerator
##img_dat_gen
train_datagen=ImageDataGenerator(preprocessing_function=preprocess_input) #included in our dependencies
img_width, img_height = 224, 224
train_generator=train_datagen.flow_from_directory('/scratch/jw16567/TCGA-HE-Image/data/model/train/',
                                                 target_size=(img_width, img_height),
                                                 color_mode='rgb',
                                                 batch_size=32,
                                                 class_mode='categorical',
                                                 shuffle=True)

#Step3 training the model on the dataset
##train_step
model.compile(optimizer='Adam',loss='categorical_crossentropy',metrics=['accuracy'])
# Adam optimizer
# loss function will be categorical cross entropy
# evaluation metric will be accuracy

step_size_train=train_generator.n//train_generator.batch_size
model.fit_generator(generator=train_generator,
                   steps_per_epoch=step_size_train,
                   epochs=10)

model.save('/scratch/jw16567/TCGA-HE-Image/data/model/my_model.h5')

print train_generator.class_indices
P_count=0
Pimg_path="/scratch/jw16567/TCGA-HE-Image/data/model/test/Enriched/"
for f in os.listdir(Pimg_path):
	img = image.load_img(Pimg_path+f, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#print('Input image shape:', x.shape)
	preds = str(model.predict(x).argmax(axis=-1))[1]
	if preds=='1':
		P_count=P_count+1


Nimg_path="/scratch/jw16567/TCGA-HE-Image/data/model/test/Depleted/"
N_count=0
for f in os.listdir(Nimg_path):
	img = image.load_img(Nimg_path+f, target_size=(img_width, img_height))
	x = image.img_to_array(img)
	x = np.expand_dims(x, axis=0)
	x = preprocess_input(x)
	#print('Input image shape:', x.shape)
	preds = str(model.predict(x).argmax(axis=-1))[1]
	if preds=='0':
		N_count=N_count+1
print('Accuracy: '+str(float(P_count+N_count)/float(len(os.listdir(Pimg_path))+len(os.listdir(Nimg_path)))))
