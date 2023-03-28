import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image


target_folder="All_target_masks"
pred_folder="clustered_dataset"


target_images=[]

for i in os.listdir(target_folder):
	k=os.path.join(target_folder,i)
	t_img = np.asarray(Image.open(k))
	target_images.append(t_img)
print(target_images)
	
print(len(target_images))
print("**********************************************************")	
pred_images=[]
for i in os.listdir(pred_folder):
	k=os.path.join(pred_folder,i)
	pimg = np.asarray(Image.open(k))
	gray = cv2.cvtColor(pimg, cv2.COLOR_BGR2GRAY)
	ret,threshp = cv2.threshold(gray,70,255,0)   #binary masks of clustered imag
	img = Image.fromarray(threshp)
	img.save(f'{pred_folder}/bin_{i}.jpg')
	#pred_mask = np.pad(pred_mask, ((0, 310-112), (0, 300-142)), mode='constant', constant_values=0)
	
#print((pred_images))

result=[]
for i in range(len(pred_images)):
	intersection = np.logical_and(pred_images[i], target_images[i]).sum()
	union = np.logical_or(pred_images[i], target_images[i]).sum()
	jaccard = intersection / union
	result.append(jaccard)
	print("iou",jaccard)
	with open ('writeme.txt', 'w') as file:
		file.write('writeme')
		




# img = np.asarray(Image.open('predicted.jpg'))
# print(img.shape)
# target_mask = np.asarray(Image.open('target.jpg'))
# print(target_mask.shape)   #(112,142)
# #print(pred_mask.shape,target_mask.shape)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #310,300
# print(gray.shape)
# # apply thresholding to convert grayscale to binary image
# ret,thresh = cv2.threshold(gray,70,255,0)
# print(thresh.shape[0])  #310,300

# target_mask = np.pad(target_mask, ((0, 310-112), (0, 300-142)), mode='constant', constant_values=0)
# print(target_mask.shape)
                                    #(0,img.shape[0]-target_mask.shape[0]),(0,img.shape[1]-target_mask.shape[1])








