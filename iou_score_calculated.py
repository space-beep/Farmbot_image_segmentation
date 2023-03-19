import cv2
import numpy as np
from matplotlib import pyplot as plt
import os
from os import listdir 
import argparse
import sys

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from PIL import Image
from sklearn import preprocessing
from sklearn.metrics.pairwise import euclidean_distances

#Create a mask

low_green=np.array([25,52,72])
high_green=np.array([102,255,255])


#Importing dataset Ara2013RPi with 166 images
#folder_Dir="Plant_Phenotyping_Datasets/Plant_Phenotyping_Datasets/Plant/Ara2012"
folder_Dir="Ara2012"
#print(os.listdir(folder_Dir))
cnt=0
my_array = os.listdir(folder_Dir)
#print(my_array)
#my_array = my_array[:2]
#print(my_array)

iou_score_list=[]
good_score=0

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
        
        img_label_bw = np.sum(img_label,axis=2)

        img_label_bw = (img_label_bw>0)
        
        #IoU calculation
        intersection = np.logical_and(img_label_bw, mask)
        union = np.logical_or(img_label_bw, mask)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        if iou_score>0.7:
            good_score=1
            iou_score_list.append(good_score)
print(len(iou_score_list))       
