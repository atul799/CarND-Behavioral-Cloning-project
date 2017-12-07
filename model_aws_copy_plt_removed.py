# -*- coding: utf-8 -*-
"""

@author: atpandey
"""

#%%
from os import getcwd
import random
import csv
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import tensorflow as tf

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.regularizers import l2
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle




#################################
#%%

#this section reads in different sources of data
#adds left cam right cam and steering angle
#then randomly discards angles close to 0 degree


# dataset types

my_data = True
udacity_data = True
track2_data =True

data_to_use = [my_data, udacity_data,track2_data]
img_path_prepend = [getcwd() + '/my_data/data/', getcwd() + '/data_from_udacity/data/',getcwd() + '/mountain_track_data/data/']
csv_path = [getcwd() + '/my_data/data/driving_log.csv', getcwd() + '/data_from_udacity/data/driving_log.csv',getcwd() + '/mountain_track_data/data//driving_log.csv']


#image (left center and right) path from csv file and steering angles
image_paths = []
angles = []


#iterate througgh all data sources
for j in range(3):
    
    
    if not data_to_use[j]:
        # 0 = localdata, 1 = Udacity data,2 mountain track
        print('not using dataset: ', j)
        continue
    # Read csv files,skip first line and store
    #left center and right image paths in image_paths
    #store steering angle in angles
    with open(csv_path[j], newline='') as f:
        driving_data = list(csv.reader(f, skipinitialspace=True, delimiter=','))

    # for data from csv files append to img_paths and steering angles
    #right cam -0.2 adjustment
    #left cam +0.2 adjustment
    for row in driving_data[1:]:
        # skip low speed samples
        if float(row[6]) < 0.1 :
            continue
        # center image 
        image_paths.append(img_path_prepend[j] + row[0])
        angles.append(float(row[3]))
        # left image 
        image_paths.append(img_path_prepend[j] + row[1])
        angles.append(float(row[3])+0.2)
        # right image 
        image_paths.append(img_path_prepend[j] + row[2])
        angles.append(float(row[3])-0.2)

#
image_paths = np.array(image_paths)
angles = np.array(angles)

print('Raw Samples:', image_paths.shape, angles.shape)
#%%
# Analyze data

num_bins=20
avg_samples_per_bin = len(angles)/num_bins
print("average number of samples per bins:",avg_samples_per_bin)
histpre, binspre = np.histogram(angles, num_bins)
width = 0.7 * (binspre[1] - binspre[0])
center = (binspre[:-1] + binspre[1:]) / 2
#removed for aws
#figpre,axspre=plt.subplots()
#axspre.bar(center, histpre, align='center', width=width)
##axspre.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#axspre.set_title("pre angle distribuiton")
#axspre.grid()


# determine keep probability for each bin: 
#if below avg_samples_per_bin, keep all; otherwise keep prob is proportional
# to number of samples above the average, so as to bring the number of samples 
#for that bin down to the average
keep_probs = []
target = avg_samples_per_bin 
for i in range(num_bins):
    if histpre[i] < target:
        keep_probs.append(1.)
    else:
        keep_probs.append(1./(histpre[i]/target))
remove_list = []
for i in range(len(angles)):
    for j in range(num_bins):
        if angles[i] > binspre[j] and angles[i] <= binspre[j+1]:
            # delete from X and y with probability 1 - keep_probs[j]
            if np.random.rand() > keep_probs[j]:
                remove_list.append(i)
                
#drop data samples
image_paths = np.delete(image_paths, remove_list, axis=0)
angles = np.delete(angles, remove_list)

# print histogram again to show more even distribution of steering angles
hist, bins = np.histogram(angles, num_bins)
#removed for aws
#figpost,axspost=plt.subplots()
#axspost.bar(center, hist, align='center', width=width)
##axspost.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#axspost.set_title("post angle distribuiton")
#axspost.grid()
##plt.show()

avg_samples_per_bin_post = len(angles)/num_bins
print("average number of samples per bins post drop:",avg_samples_per_bin_post)
print('Samples for training:', image_paths.shape, angles.shape)
#%%
#Augmentation and transform schemes
def crop_image(img,top=50,bottom=140):
    '''
    Method for cropping image 
    '''
    #Crop image
    return img[top:bottom,:,:]
    
def yuv_image(img,colorspace=cv2.COLOR_BGR2YUV):
    '''
    convert to YUV space 
    (in drive.py, use cv2.COLOR_RGB2YUV)
    '''
    # YUV color space (is used in nvidia paper
    yuv_img = cv2.cvtColor(img, colorspace)
    #yuv_img=yuv_img/255. - 0.5
    return yuv_img
def resize_image(img,col=66,row=200,interpolationin = cv2.INTER_AREA):
    '''
    resize per nvidia paper
    '''
    # resize per nvidia paper to 66x200x3 
    return cv2.resize(img,(row, col), interpolation = interpolationin)


def intensity_image(img):
    '''
    add random intensity to pixels by converting to HSV space
    '''
    hsv=cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    intensity=np.random.random()
    if intensity > 0.9:
       hsv[:,:,2]=0.95*hsv[:,:,2]
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2RGB)

def equalize_image(img):
    '''
    Apply histogram equalization
    '''
    yuv=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    yuv[:,:,0]=cv2.equalizeHist(yuv[:,:,0])
    return cv2.cvtColor(yuv,cv2.COLOR_YUV2RGB)


def random_bright(img):
    ''' 
    method for random brightness adjust 
    '''
    #new_img = img.astype(float)
    new_img=cv2.cvtColor(img,cv2.COLOR_RGB2YUV)
    # random brightness - the mask bit keeps values from going beyond (0,255)
    value = np.random.randint(0, 60)
    if value > 30:
        mask = (new_img[:,:,0] + value) > 255 
    if value <= 30:
        mask = (new_img[:,:,0] - value) < 0
    new_img[:,:,0] += np.where(mask, 0, value).astype(np.uint8)
    
 
    return cv2.cvtColor(new_img,cv2.COLOR_YUV2RGB)

def random_shift(img, angle):
    ''' 
    method for horiz affine translation  
    '''
#    # randomly shift 
    h,w,_ = img.shape
    delx = np.random.randint(50)
    dely = np.random.randint(20)
    M = np.float32([[1, 0, delx], [0, 1, dely]])
    im_af = cv2.warpAffine(img, M, (w, h))
    angle_af = angle + (float(delx) * 0.001)

    return (im_af.astype(np.uint8), angle_af)
###############################
idx=np.random.randint(len(image_paths))
imgt=image_paths[idx]
print("image path is",imgt, 'at index',idx)
img = cv2.imread(imgt)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
brightened_img=random_bright(img)
transed_image,trans_angle=random_shift(img,angles[idx])
cropped_image=crop_image(img,top=50,bottom=140)
yuved_image=yuv_image(img)
intensity_imageo=intensity_image(img)
equalize_imageo=equalize_image(img)
resize_imageo=resize_image(img)
#removed for aws
#figs,axss=plt.subplots(2,4)
#axss=axss.ravel()
#axss[0].imshow(img,aspect='auto')
#axss[0].set_title("orig picture")
#axss[1].imshow(brightened_img,aspect='auto')
#axss[1].set_title("random brightness change")
#axss[2].imshow(transed_image,aspect='auto')
#axss[2].set_title("random shift")
print("angle translated from:",angles[idx],"to:",trans_angle)
#axss[3].imshow(cropped_image,aspect='auto')
#axss[3].set_title("cropped image")
#axss[4].imshow(yuved_image,aspect='auto')
#axss[4].set_title("yuv image")
#axss[5].imshow(intensity_imageo,aspect='auto')
#axss[5].set_title("intensity changed")
#axss[6].imshow(equalize_imageo,aspect='auto')
#axss[6].set_title("histogram equalization applied")
#axss[7].imshow(resize_imageo,aspect='auto')
#axss[7].set_title("resized image")

#%%
# split into train/test sets, since lots of 0 angles are already dropped validation set is set at smaller value
image_paths_train, image_paths_valid, angles_train, angles_valid = train_test_split(image_paths, angles,
                                                                                  test_size=0.1)
print('Train:', image_paths_train.shape, angles_train.shape)
print('Valid:', image_paths_valid.shape, angles_valid.shape)


#%%
#Generators used to reduce memory requirement
#angles used in training
angles_picked=[]
flip_angles_picked=[]

def generate_training_data(image_paths, angles, batch_size=128, valid=False,top=50,bottom=140):
    '''
    if valid then image is not adjusted for brightness or affine translated
    random flip added for angles gt 0.25,
    '''
    image_paths, angles = shuffle(image_paths, angles)
    Xt,yt = ([],[])
    while True:       
        for idx in range(len(angles)):
            img = cv2.imread(image_paths[idx])
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            #print("img path:",image_paths[idx],"imgshape:",img.shape)
            angle = angles[idx]            
            img=crop_image(img)
            #img=yuv_image(img)
            #intensity change
            img=intensity_image(img)
            #img=random_bright(img)
            #histogram equalization
            #img=equalize_image(img)
            if not valid:
                flip_coin = random.randint(0,3)
                if flip_coin==1:
                    img=random_bright(img)
                elif flip_coin==2:
                    img,angle=random_shift(img,angle)
                elif flip_coin ==3:
                    img=equalize_image(img)
            Xt.append(img)
            yt.append(angle)
            angles_picked.append(angle)
            if len(Xt) == batch_size:
                yield (np.array(Xt), np.array(yt))
                Xt, yt = ([],[])
                image_paths, angles = shuffle(image_paths, angles)
            # flip image
            if abs(angle) > 0.25:
                img = cv2.flip(img, 1)
                angle *= -1
                Xt.append(img)
                yt.append(angle)
                flip_angles_picked.append(angle)
                if len(Xt) == batch_size:
                    yield (np.array(Xt), np.array(yt))
                    Xt, yt = ([],[])
                    image_paths, angles = shuffle(image_paths, angles)




#generatror functions for train and valid dataset

train_generator = generate_training_data(image_paths_train, angles_train, valid=False)
valid_generator =   generate_training_data(image_paths_valid, angles_valid, valid=True)


#%%
#Nvidia architecture
model = Sequential()

# Normalize

model.add(Lambda(lambda x: x/255. - 0.5,input_shape=(90,320,3)))


#check if weight plotting works by using 1x1 kernel so original image will be reproduced in the plot
#model.add(Convolution2D(24, 1, 1))
#model.add(ELU())


model.add(Convolution2D(24, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), border_mode='valid',W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Convolution2D(64, 3, 3, border_mode='valid',W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Convolution2D(64, 3, 3, border_mode='valid',W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))


# Flatten layer
model.add(Flatten())
#FC layers
model.add(Dense(100,W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(50,W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))
model.add(Dense(10,W_regularizer=l2(0.001)))
model.add(BatchNormalization())
model.add(ELU())
#model.add(Dropout(0.50))

#FC layer for output
model.add(Dense(1))

# Compile and train the model, 
#model.compile('adam', 'mean_squared_error')
model.compile(optimizer=Adam(lr=0.0001), loss='mse')

###################################
#%%
# Checkpoints configuration
checkpoint_file = 'nvidia.chk'
checkpoint = ModelCheckpoint(checkpoint_file, monitor='val_loss', verbose=0, save_best_only=False, mode='min')
callbacks = [checkpoint]


history = model.fit_generator(train_generator,samples_per_epoch=2*len(angles_train), validation_data=valid_generator, nb_val_samples=2*len(angles_valid), 
                              nb_epoch=20, callbacks=[checkpoint])


print(model.summary())


# Save model data
model.save_weights('./modelw.h5')
model.save('./model.h5')
print("number of angles list used in trg:",len(angles_picked))
print("flipped angles picked:",len(flip_angles_picked))

#############
total_angles_in_trg=angles_picked+flip_angles_picked
histend, binsend = np.histogram(total_angles_in_trg, num_bins)
#removed for aws
#figend,axsend=plt.subplots()
#axsend.bar(center, histend, align='center', width=width)
##axspost.plot((np.min(angles), np.max(angles)), (avg_samples_per_bin, avg_samples_per_bin), 'k-')
#axsend.set_title("End angle distributon")
#axsend.grid()i
##plt.show()

avg_samples_per_bin_end = len(total_angles_in_trg)/num_bins
print("average number of samples per bins post training:",avg_samples_per_bin_end)
#%%
#%%
from keras import backend as K
def layer_to_visualize(layer):
    inputs = [K.learning_phase()] + model.inputs
    print("input_shape",len(inputs))
    _convout1_f = K.function(inputs, [layer.output])
    
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(img_to_visualize)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)
    
    n = convolutions.shape[2]
    n = int(np.ceil(np.sqrt(n)))
    
    # Visualization of each filter of the layer
    #removed for aws
    #fig = plt.figure(figsize=(12,8))
    #for i in range(len(convolutions)):
    #    ax = fig.add_subplot(n,n,i+1)
    #    #ax.imshow(convolutions[i], cmap='gray',aspect='auto')
    #    ax.imshow(convolutions[...,i],aspect='auto',cmap='gray')



rnd =  np.random.randint(len(image_paths_train))
#rnd = 8500
row = image_paths_train[rnd]


img = cv2.imread(image_paths[rnd])
#print("img path:",image_paths[idx],"imgshape:",img.shape)
angle = angles[idx]

img=crop_image(img)
#img=yuv_image(img)    

       
img_to_visualize=img
#removed for aws
#fig1,axs1=plt.subplots()
#axs1.imshow(img_to_visualize)
#axs1.set_title("base image")
print("Done")
# Keras requires the image to be in 4D
# So we add an extra dimension to it.
img_to_visualize = np.expand_dims(img_to_visualize, axis=0)

# Specify the layer to want to visualize
layer_to_visualize(model.layers[1])

# As convout2 is the result of a MaxPool2D layer
# We can see that the image has blurred since
# the resolution has reduced 
#layer_to_visualize(model.layers[3])
