import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image

#folder_dir="binary_outputs_clustered"

target_folder="All_target_masks"
pred_folder="binary_ouputs_clustered"




pred_images=[]
for i in os.listdir(pred_folder):
	k=os.path.join(pred_folder,i)
	pred_mask  = np.asarray(Image.open(k))
	pred_images.append(pred_mask)
print(len(pred_images))   #156 images only

print("**********************************************************")	

target_images=[]
cnt=0
for i in os.listdir(target_folder):
	k=os.path.join(target_folder,i)
	target_mask = np.asarray(Image.open(k))
	target_mask=np.pad(target_mask, ((0,abs(pred_images[cnt].shape[0]-target_mask.shape[0])),(0,abs(pred_images[cnt].shape[1]-target_mask.shape[1]))), mode='constant', constant_values=0)
	cnt+=1
	target_images.append(target_mask)

all_accuracies=[]
all_recalls=[]
all_precisions=[]
result=[]  #IOUS
for i in range(len(target_images)):
	intersection = np.logical_and(pred_images[i], target_images[i]).sum()
	union = np.logical_or(pred_images[i], target_images[i]).sum()
	jaccard = intersection / union
	result.append(jaccard)
	print("iou",jaccard)
	result.append(jaccard)
	print(np.mean(result))
	correct_pixels = np.sum(pred_images[i] == target_images[i])
	total_pixels = pred_images[i].shape[0] * pred_images[i].shape[1]
	accuracy = correct_pixels*100 / total_pixels
	print("Accuracy is ",accuracy)
	all_accuracies.append(accuracy)
	true_positives = np.logical_and(pred_images[i], target_images[i]).sum()
	false_negatives = np.logical_and(np.logical_not(pred_images[i]), target_images[i]).sum()
	recall = true_positives / (true_positives + false_negatives)
	all_recalls.append(recall)
	print(recall)

	#PRECISION
	true_positives = np.logical_and(pred_images[i], target_images[i]).sum()
	false_positives = np.logical_and(pred_images[i], np.logical_not(target_images[i])).sum()
	precision = true_positives / (true_positives + false_positives)
	all_precisions.append(precision)
	print("Average of all accuracies:",np.mean(all_accuracies))
	print("Average of all recalls:",np.mean(all_recalls))
	print("Average of all precisions:",np.mean(all_precisions))

#with open ('writeme.txt', 'w') as file:
#	file.write('writeme')
#  FINAL MEAN = 0.11453825626313556


