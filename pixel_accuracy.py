import numpy as np
from PIL import Image
from precision_score import precision
from recall_score import recall
from jaccard_index import jaccard_index

def pixel_accuracy(pred, target):
    """
    Computes pixel accuracy between two images.
    Args:
        pred (str): path to predicted image.
        target (str): path to target image.
    Returns:
        accuracy (float): pixel accuracy between the images.
    """
    # Load images as numpy arrays
    pred_image = np.asarray(Image.open(pred))
    target_image = np.asarray(Image.open(target))
    
    # Check if the images have the same size
    if pred_image.shape != target_image.shape:
        raise ValueError("Images must have the same dimensions.")
    
    # Calculate accuracy
    correct_pixels = np.sum(pred_image == target_image)
    total_pixels = pred_image.shape[0] * pred_image.shape[1]
    accuracy = correct_pixels*100 / total_pixels
    
    return accuracy
    
accuracy = pixel_accuracy('predicted.png', 'original.png')

jaccard = jaccard_index('predicted.png', 'original.png')
precision = precision('predicted.png', 'original.png')
recall = recall('predicted.png', 'original.png')


print("Jaccard Index:", jaccard)
print("Pixel accuracy:", accuracy)
print("Precision :",precision)
print("Recall score :",recall)

