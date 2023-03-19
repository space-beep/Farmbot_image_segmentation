import numpy as np
from PIL import Image

def jaccard_index(pred, target):
    """
    Computes the Jaccard Index (Intersection over Union) between two images.
    Args:
        pred (str): path to predicted image.
        target (str): path to target image.
    Returns:
        jaccard (float): Jaccard Index between the images.
    """
    # Load images as numpy arrays
    pred_mask = np.asarray(Image.open(pred))
    target_mask = np.asarray(Image.open(target))
    
    # Check if the masks have the same size
    if pred_mask.shape != target_mask.shape:
        raise ValueError("Masks must have the same dimensions.")
    
    # Calculate Jaccard Index
    intersection = np.logical_and(pred_mask, target_mask).sum()
    union = np.logical_or(pred_mask, target_mask).sum()
    jaccard = intersection / union
    
    return jaccard

