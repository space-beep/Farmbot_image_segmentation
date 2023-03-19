import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir 
from shutil import copy2
from sklearn.cluster import DBSCAN

folder_Dir="Ara2012"

my_array = os.listdir(folder_Dir)
iou_score_list=[]
good_score=0
output_folder = 'output'

for i in my_array:
    if i[-8:]=='_rgb.png':
        mask_name = i
        mask_name = mask_name.replace('_rgb.png','_label.png')
        
        k=os.path.join(folder_Dir,i)
        mask_name=os.path.join(folder_Dir,mask_name)
        #print(k) #file name rgb_blah_blah.jpg
        #print(mask_name)
        img=cv2.imread(k)
        img_label=cv2.imread(mask_name)
        img_arr_label = np.asarray(img_label)
        img_label_bw = np.sum(img_label,axis=2)
        img_label_bw = (img_label_bw>0)
        
        
        img2=img.reshape((-1,3))
        dbscan=DBSCAN(eps=0.005,min_samples=500)
        dbscan.fit(img2)
        print("length",len(dbscan.core_sample_indices_))
        unique_labels=np.unique(dbscan.labels_)
        for label in unique_labels:
            idx=np.min(np.argwhere(dbscan.labels_==label))
            img2[dbscan.labels_==label]=img2[idx]
        res2=img2.reshape(img.shape)
        #print(res2)
        cv2.imshow('clustered iamge', res2)
        basename = os.path.basename(i)
        name = os.path.splitext(basename)[0]
        cv2.imwrite(folder_Dir + name + '_dbscan_seg.jpg', res2)
        img_arr_label2 = np.asarray(res2)
        img_label_bw2 = np.sum(res2,axis=2)
        img_label_bw2 = (img_label_bw2>0)
        intersection = np.logical_and(img_label_bw, img_label_bw2)
        union = np.logical_or(img_label_bw, img_label_bw2)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        if iou_score>0.7:
            good_score=1
            iou_score_list.append(good_score)
        
        
print(len(iou_score_list)) 
        #print(res2) home/krupal/Documents/GITCLONEDSTUFF/kmeans2/outputkmeans



