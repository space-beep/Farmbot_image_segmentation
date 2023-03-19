import numpy as np
from PIL import Image

def recall(pred, target):
    """
    Computes the Recall of the predicted image segmentation.
    Args:
        pred (str): path to predicted image.
        target (str): path to target image.
    Returns:
        recall (float): Recall of the predicted segmentation.
    """
    # Load images as numpy arrays
    pred_mask = np.asarray(Image.open(pred))
    target_mask = np.asarray(Image.open(target))
    
    # Check if the masks have the same size
    if pred_mask.shape != target_mask.shape:
        raise ValueError("Masks must have the same dimensions.")
    
    # Calculate Recall
    true_positives = np.logical_and(pred_mask, target_mask).sum()
    false_negatives = np.logical_and(np.logical_not(pred_mask), target_mask).sum()
    recall = true_positives / (true_positives + false_negatives)
    
    return recall

