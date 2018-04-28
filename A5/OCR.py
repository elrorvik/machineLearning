from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import numpy as np
import skimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.color import rgb2gray

def label_to_letter(label):
    return chr(label+97)

def letter_to_label(letter):
    return int(ord(letter)-97)

def get_image(folder_path):
    col = imread_collection(folder_path) #creating a collection with the available images
    train = concatenate_images(col)
    label = np.zeros((train.shape[0],1))

    for (file_name,i) in zip(col.files,range(train.shape[0])):
        label[i] = letter_to_label(file_name[14])
    return label,train

def split_train_test(y,X):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    return X_train,X_test,y_train,y_test


col_dir = 'chars74k-lite/*/*.jpg'
label,image = get_image(col_dir)

train_images,test_images,train_labels,test_labels = split_train_test(label,image)

#print(train_images.shape)
#for i in range(train_images.shape[0]):
#    train_images[i] = rgb2gray(train_images[i])

#df= hog(train_images, orientations=8, pixels_per_cell=(5,5), cells_per_block=(2, 2))


plt.figure()
plt.imshow(train_images[4000])
plt.figure()
plt.imshow(train_images[10])
print(label_to_letter(train_labels[4000]))
print(label_to_letter(train_labels[10]))
plt.show()
