#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 16:45:21 2018

@author: giovannagraciani
File (3/3)

This file reads in a .h5 files from a trained model to generate 
predictions for test data.
"""

from keras.models import load_model
from keras.preprocessing import image
from keras.utils import plot_model
import os, os.path
import numpy as np
import pandas as pd
import sys

#Arrays
image_path_list = []
valid_image_extensions = [".jpg"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
file_names = []

#bring in path list
i = sys.argv[1] 
o = sys.argv[3]
input_path = sys.argv[2] #input file path
output_path = sys.argv[4] #output file path

# dimensions of our images
img_width, img_height = 150,150

# load the model we saved
model = load_model('model_five.h5')
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

#print(model.summary())
#plot_model(model, to_file='model.png', show_shapes= True, show_layer_names=True)

#list all valid files in directory by appending valid files 
#to image_path_list
for file in os.listdir(input_path):
    name = os.path.splitext(file)[0]
    extension = os.path.splitext(file)[1]
    file_names.append(name)
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(input_path, file))

array_images = []
gender_prediction = []

#save each image in directory into an array representation 
#to pass through model for predictions
for imagePath in image_path_list:
    img = image.load_img(imagePath, target_size=(img_width, img_height))
    y = image.img_to_array(img)
    y = np.expand_dims(y, axis=0)
    array_images.append(image)
    genderPrediction = model.predict(y)
    gender_prediction.append(genderPrediction)


#save userid and their corresponding predictions
df2 = pd.DataFrame({'userid':file_names, 'prediction':gender_prediction})

#round since 0 - F, 1 - M and prediction will be a probability between 0-1
df2['prediction'] = df2['prediction'].apply(np.round)
df2['prediction'] = df2['prediction'].astype(str)
df2['prediction'] = ['0.0' if '0.' in x else '1.0' for x in df2['prediction']]

print(df2)
'''
#export prediction to xml files
for index, row in df2.iterrows():
    #print(row['userid'], row['prediction'])
    userid = row['userid']
    if row['prediction'] == '[[0.]]':
        gender = 'gender=\"female\"\n' 
        #print(gender)
    else:
        gender = 'gender=\"male\"\n' 
        #print(gender)
    
    file = open(output_path +"/" +userid + ".xml", "w") 
    file.write("<user\n")
    file.write("\tid=\"")
    file.write(userid)
    file.write("\"\n")
    file.write("age_group=\"xx-24\"\n")
    file.write(gender)
    file.write("extrovert=\"3.49\"\n")
    file.write("neurotic=\"2.73\"\n")
    file.write("agreeable=\"3.58\"\n")
    file.write("conscientious=\"3.45\"\n")
    file.write("open=\"3.91\"\n")
    file.write("/>")
    file.close()
'''
