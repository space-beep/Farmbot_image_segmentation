import numpy as np
from PIL import Image

def precision(pred, target):
    """
    Computes the Precision of the predicted image segmentation.
    Args:
        pred (str): path to predicted image.
        target (str): path to target image.
    Returns:
        precision (float): Precision of the predicted segmentation.
    """
    # Load images as numpy arrays
    pred_mask = np.asarray(Image.open(pred))
    target_mask = np.asarray(Image.open(target))
    
    # Check if the masks have the same size
    if pred_mask.shape != target_mask.shape:
        raise ValueError("Masks must have the same dimensions.")
    
    # Calculate Precision
    true_positives = np.logical_and(pred_mask, target_mask).sum()
    false_positives = np.logical_and(pred_mask, np.logical_not(target_mask)).sum()
    precision = true_positives / (true_positives + false_positives)
    
    return precision

