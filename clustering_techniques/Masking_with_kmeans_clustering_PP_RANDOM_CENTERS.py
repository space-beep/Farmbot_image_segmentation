#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
from os import listdir 


# In[10]:


#img=cv2.imread('plant.jpg')


# In[5]:


#img.shape

#GBR OF IMAGE (color space is GBR in cv2)


# In[7]:


#flatten this image for kmeans

#img2=img.reshape((-1,3))


# In[8]:


#img2.shape


# In[9]:


#type(img2)


# In[26]:


folder_Dir="Ara2012"
my_array = os.listdir(folder_Dir)
iou_score_list=[]
good_score=0
all_ious=[]
all_accuracies=[]
all_recalls=[]
all_precisions=[]
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
        #print("image label cv2 shape",img_label.shape)
        img_arr_label = np.asarray(img_label)
        #print("img array label", img_arr_label.shape)
        img_label_bw = np.sum(img_label,axis=2)
        img_label_bw = (img_label_bw>0)
        myreshapedlabel=img_label_bw.reshape(-1,1)
        #print("image label reshaped: ",myreshapedlabel.shape)
        plt.figure("myreshapedlabel")
        plt.imshow(img_label)
        plt.show()

        
        img2=img.reshape((-1,3))
        #print("img2 reshaped: ",img2)
        Z=img2
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 2
        attempts=10
        #Now this next line returns 3 things
        #1.Compactness: sum of sqd. distance from each point to thier corresponging centers
        #2.Label
        #3. Centers
        ret,label,center=cv2.kmeans(Z,K,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)#different criteria can be used cv2.KMEANS_RANDOM_CENTERS,cv.2KMEANS_PP_CENTERS
        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        #print(center)
        #print("kmeans label shape",label.shape)
        res = center[label.flatten()]
        #print("res shape",res.shape)
        res2 = res.reshape((img.shape))
#         print(res2.shape)
        plt.figure("img_segmented")
        plt.imshow(res2)
        plt.show()


        #IoU calculation
        intersection = np.logical_and(myreshapedlabel, label)
        union = np.logical_or(myreshapedlabel, label)
        iou_score = np.sum(intersection) / np.sum(union)
        print("IoU is %s" % iou_score)
        if iou_score>0.7:
            good_score=1
            iou_score_list.append(good_score)
        all_ious.append(iou_score)
        
        #ACCURACY
        correct_pixels = np.sum(label == myreshapedlabel)
        total_pixels = label.shape[0] * label.shape[1]
        accuracy = correct_pixels*100 / total_pixels
        print("Accuracy is ",accuracy)
        all_accuracies.append(accuracy)
        
        #RECALL
        true_positives = np.logical_and(label, myreshapedlabel).sum()
        false_negatives = np.logical_and(np.logical_not(label), myreshapedlabel).sum()
        recall = true_positives / (true_positives + false_negatives)
        all_recalls.append(recall)
        print(recall)
        
        #PRECISION
        true_positives = np.logical_and(label, myreshapedlabel).sum()
        false_positives = np.logical_and(label, np.logical_not(myreshapedlabel)).sum()
        precision = true_positives / (true_positives + false_positives)
        all_precisions.append(precision)


# In[12]:


print("Average of all ious:",np.mean(all_ious))


# In[13]:


print("Average of all accuracies:",np.mean(all_accuracies))


# In[14]:


print("Average of all recalls:",np.mean(all_recalls))


# In[15]:


print("Average of all precisions:",np.mean(all_precisions))


print(len(iou_score_list)) 


# In[ ]:




