#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir 
import argparse
import sys
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances


# In[11]:


#Create a mask

low_green=np.array([25,52,72])
high_green=np.array([102,255,255])
iterations = 5
#img_arr = np.asarray(img)
k='2'

all_ious=[]
all_accuracies=[]
all_recalls=[]
all_precisions=[]
#Importing dataset Ara2013RPi with 166 images
#folder_Dir="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2012"
folder_Dir="Ara2012"
#print(os.listdir(folder_Dir))
cnt=0
my_array = os.listdir(folder_Dir)
#print(my_array)
#my_array = my_array[:2]
#print(my_array)
for i in my_array:
    if i[-8:]=='_rgb.png':
        mask_name = i
        mask_name = mask_name.replace('_rgb.png','_label.png')
        cnt +=1
        k=os.path.join(folder_Dir,i)
        mask_name=os.path.join(folder_Dir,mask_name)
        #print(k) #file name rgb_blah_blah.jpg
        #print(mask_name)
        img=cv2.imread(k)
        img_label=cv2.imread(mask_name)
        img_arr = np.asarray(img)
        img_arr_label = np.asarray(img_label)
        
        img_color=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        #img_label_color=cv2.cvtColor(img_label,cv2.COLOR_BGR2HSV)
        
        mask=cv2.inRange(img_color,low_green,high_green)
        print("computed mask is",mask)
        plt.figure("Computed Masked image")
        plt.imshow(mask)
        plt.show()
        print(mask.shape)

        #img_label=cv2.cvtColor(img_label,cv2.COLOR_BGR2GRAY)
        print("max")
        print(np.max(img_arr_label))
        print("min")
        print(np.min(img_arr_label))
        print(img_label.shape)
        img_label_bw = np.sum(img_label,axis=2)
#         if (img_label_bw).any()>0:
#             img_label_bw=1
#         else:
#             img_label_bw=0
        img_label_bw = (img_label_bw>0)
        print(img_label_bw.shape)
        print("Actual (true) mask is",img_label_bw)
        plt.figure("Actual (true) masked image")
        plt.figure("img_label")
        plt.imshow(img_label_bw)
        plt.show()
        print(img_label_bw.shape)
        #IoU calculation
        intersection = np.logical_and(img_label_bw, mask)
        union = np.logical_or(img_label_bw, mask)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        all_ious.append(iou_score)
        correct_pixels = np.sum(img_label_bw == mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        accuracy = correct_pixels*100 / total_pixels
        print("Accuracy is ",accuracy)
        all_accuracies.append(accuracy)
        true_positives = np.logical_and(img_label_bw, mask).sum()
        false_negatives = np.logical_and(np.logical_not(mask), img_label_bw).sum()
        recall = true_positives / (true_positives + false_negatives)
        all_recalls.append(recall)
        print(recall)
        true_positives = np.logical_and(mask, img_label_bw).sum()
        false_positives = np.logical_and(mask, np.logical_not(img_label_bw)).sum()
        precision = true_positives / (true_positives + false_positives)
        all_precisions.append(precision)


# In[12]:


print(np.mean(all_ious))


# In[13]:


print(np.mean(all_accuracies))


# In[14]:


print(np.mean(all_recalls))


# In[15]:


print(np.mean(all_precisions))


# In[ ]:




