from skimage.io import imread_collection, imshow
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

#your path
col_dir = 'chars74k-lite/*/*.jpg'

#creating a collection with the available images
col = imread_collection(col_dir)
print(len(col))
train = np.zeros((len(col), col[0].shape[0], col[0].shape[1]))
#print(col.shape)
print(col[0].shape)
plt.figure()
plt.imshow(col[-1], cmap=plt.cm.gray)
plt.show()
