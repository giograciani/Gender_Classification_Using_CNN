#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 12:28:48 2018

@author: giovannagraciani
File (1/3)

This file will read in all images from a folder, move each file
to it's appropriate directory based on the user's gender. Having 
sorted training data enables us to train a CNN for the task of binary
classification in ImageTrain.py
"""

import os, os.path
from tqdm import tqdm
import pandas as pd
import shutil

#Arrays
image_path_list = []
valid_image_extensions = [".jpg"]
valid_image_extensions = [item.lower() for item in valid_image_extensions]
file_names = []
array_images = []
path_list = []

#Directories
imageDir = "projectData/training/image"
male_path = '/Users/giovannagraciani/Desktop/455/train/male/'
female_path = '/Users/giovannagraciani/Desktop/455/train/female/'
filedir='/Users/giovannagraciani/Desktop/455/projectData/training/image/'

#--- Get User data to train on ---
ud = pd.read_csv('/Users/giovannagraciani/Desktop/455/projectData/training/profile/profile.csv', index_col=0)
ud.drop(['ope','neu', 'con', 'ext', 'agr'], axis=1, inplace=True)


#list all valid files in directory by appending valid files to image_path_list
for file in tqdm(os.listdir(imageDir), ascii = True, desc = "Listing files in path"):
    name = os.path.splitext(file)[0]
    extension = os.path.splitext(file)[1]
    file_names.append(name)
    if extension.lower() not in valid_image_extensions:
        continue
    image_path_list.append(os.path.join(imageDir, file))


#--- Create DF of for each image with userid and file path  --- 
df2 = pd.DataFrame({'userid':file_names, 'img_loc':image_path_list})

#--- Merge image DF with user DF to associate each image with a gender ---
result = pd.merge(ud, df2, on='userid', how='outer')

#--- Split result DF by gender --- 
male_df = pd.DataFrame(result.get(result['gender'] == 0))
female_df = pd.DataFrame(result.get(result['gender'] ==1))

#--- Move each image into its appropriate folder ---
for file in tqdm(os.listdir(filedir), ascii = True, desc = "Moving Files"):
    userid = os.path.splitext(file)[0]
    if male_df['userid'].str.contains(userid).any():
        shutil.copy(filedir+userid+'.jpg', male_path+userid+'.jpg')
    elif female_df['userid'].str.contains(userid).any():
        shutil.copy(filedir+userid+'.jpg', female_path+userid+'.jpg')
    else:
        break;
