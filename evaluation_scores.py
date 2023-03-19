from precision_score import precision
from recall_score import recall
from jaccard_index import jaccard_index
from pixel_accuracy import pixel_accuracy

accuracy = pixel_accuracy('predicted.png', 'original.png')
jaccard = jaccard_index('predicted.png', 'original.png')
precision = precision('predicted.png', 'original.png')
recall = recall('predicted.png', 'original.png')


print("Jaccard Index:", jaccard)
print("Pixel accuracy:", accuracy)
print("Precision :",precision)
print("Recall score :",recall)

