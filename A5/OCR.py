from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

#your path
col_dir = 'chars74k-lite/*/*.jpg'

#creating a collection with the available images
col = imread_collection(col_dir)
train = concatenate_images(col)
label = np.zeros((train.shape[0],1))
#for i in range(train.shape[0]):
#    label[i] =
print(type(col))
print(train.shape)
plt.figure()
plt.imshow(train[-1], cmap=plt.cm.gray)
plt.show()
