import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir 
from shutil import copy2
from sklearn import preprocessing

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
        #print(img)
                
        img_label=cv2.imread(mask_name)
        img_arr_label = np.asarray(img_label)
        img_label_bw = np.sum(img_label,axis=2)
        img_label_bw = (img_label_bw>0)
        
        
        
        img2=img.reshape((-1,3))
        #scaler = preprocessing.StandardScaler().fit(img2)
        img2=preprocessing.normalize(img2, norm='l2')
        Z=img2
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        attempts=10
        #Now this next line returns 3 things
        #1.Compactness: sum of sqd. distance from each point to thier corresponging centers
        #2.Label
        #3. Centers
        ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        #print(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        img_arr_label2 = np.asarray(res2)
        img_label_bw2 = np.sum(res2,axis=2)
        img_label_bw2 = (img_label_bw2>0)
        cv2.imshow('Gray image', res2)
        basename = os.path.basename(i)
        name = os.path.splitext(basename)[0]
        cv2.imwrite(folder_Dir + name + '_seg.jpg', res2)

        #IoU calculation
        intersection = np.logical_and(img_label_bw, img_label_bw2)
        union = np.logical_or(img_label_bw, img_label_bw2)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        if iou_score>0.7:
            good_score=1
            iou_score_list.append(good_score)
        
        
print(len(iou_score_list)) 
        #print(res2) home/krupal/Documents/GITCLONEDSTUFF/kmeans2/outputkmeans



