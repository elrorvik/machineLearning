from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import numpy as np
import matplotlib.pyplot as plt

def label_to_letter(label):
    return chr(label+97)

def letter_to_label(letter):
    return int(ord(letter)-97)

def get_image(folder_path):
    #creating a collection with the available images
    col = imread_collection(folder_path)
    train = concatenate_images(col)
    label = np.zeros((train.shape[0],1))

    for (image, file_name,i) in zip(train, col.files,range(train.shape[0])):
        label[i] = letter_to_label(file_name[14])
    return label,train



#your path
col_dir = 'chars74k-lite/*/*.jpg'
label,train = get_image(col_dir)


plt.figure()
plt.imshow(train[4000], cmap=plt.cm.gray)
plt.figure()
plt.imshow(train[2000],cmap=plt.cm.gray)
print(label_to_letter(label[4000]))
print(label_to_letter(label[2000]))
plt.show()
