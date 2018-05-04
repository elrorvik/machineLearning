from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
import sklearn
import numpy as np
import skimage
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from skimage.feature import local_binary_pattern
from skimage import data, color, exposure
from sklearn import preprocessing
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

import sklearn.svm as ssv
from  sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from scipy import ndimage


#https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea

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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=400)
    return X_train,X_test,y_train,y_test

def data_processing(images):
    for i in range(images.shape[0]):
        grey = color.rgb2gray(images[i])
        otsuThreshold = skimage.filters.threshold_otsu(grey)
        img_bw = grey > otsuThreshold
        intArr = np.array(img_bw).astype(int)
        sciImg = np.multiply(intArr,255)
        images[i] = sciImg
    return images


def hog_feature_extraction(image):
    return hog(image,orientations=10, pixels_per_cell=(4,4), cells_per_block=(2, 2) )

def local_binary_pattern_feature_extraction(image):
    lbp = local_binary_pattern(image, 150, 4, 'uniform')
    n_bins = int(lbp.max() + 1)
    df, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return df

def training(images, labels,feature_method, classification_method):
    feature_list = []
    label_list = []
    for i in range(len(images)):
        image = images[i]
        label = labels[i][0]
        df = 0
        if(feature_method == "HOG" or feature_method == "HOG+PCA" ):
            df = hog_feature_extraction(image)
        elif(feature_method == "LBP"):
            df = local_binary_pattern_feature_extraction(image)
        feature_list.append(df)
        label_list.append(label)
    features = np.array(feature_list,dtype='float')
    pp = preprocessing.StandardScaler().fit(features)
    features = pp.transform(features)
    pca = None
    if(feature_method == "HOG+PCA" ):
        pca = PCA(n_components=600)
        pca.fit(features)
        features = pca.transform(features)
    if(classification_method == "KNN"):
        model = KNeighborsClassifier(n_neighbors=8)
    elif(classification_method == "SVM"):
        model = ssv.SVC(kernel='rbf',probability=True)
    elif(classification_method == "logistic"):
        model = LogisticRegression()
    model.fit(features,label_list)
    return model,pp,pca



def test(model,pp,pca,images, labels,feature_method, classification_method):
    feature_list = []
    label_list = []
    pros_images = images
    pros_images = data_processing(pros_images)
    for i in range(len(images)):
        image = pros_images[i]
        label = labels[i][0]
        if(feature_method == "HOG" or "HOG+PCA"):
            df = hog_feature_extraction(image)
        elif(feature_method == "LBP"):
            df = local_binary_pattern_feature_extraction(image)
        df = pp.transform(np.array([df],'float64'))
        if("HOG+PCA"== feature_method):
            df = pca.transform(df.reshape((1,-1)))
        feature_list.append(df)
        label_list.append(label)
    predict_array = []
    probs_array = []
    probs = 0
    for i in range(len(feature_list)):
        predict = model.predict(feature_list[i].reshape((1,-1)))
        probs = model.predict_proba(feature_list[i].reshape((1,-1)))
        predict_array.append(predict[0])
        probs_array.append(probs[0][int(predict)])
        print(label_to_letter(int(predict))+ ": " + str(probs[0][int(predict)]))
        print("correct: " + label_to_letter(int(label_list[i])))
        print(probs[0])
    correct = 0
    for i in range(len(predict_array)):
        if (predict_array[i] == label_list[i]):
            correct +=1

    print("accuracy" , correct/len(predict_array))
    return predict_array, probs_array


def prection_of_each_letter_in_test(true_label,pre_label, classifier):
    print(sklearn.metrics.confusion_matrix(true_label, pre_label))
    print("With the "+ classifier + " classifier")
    for i in range(len(sklearn.metrics.confusion_matrix(true_label, pre_label))):
        information =""
        information += label_to_letter(i)+": "
        information += str("{0:.2f}".format(sklearn.metrics.confusion_matrix(true_label, pre_label)[i][i]/\
        sum(sklearn.metrics.confusion_matrix(true_label, pre_label)[i])))
        information += "  " + str(sklearn.metrics.confusion_matrix(true_label, pre_label)[i][i])
        information += " of " + str(sum(sklearn.metrics.confusion_matrix(true_label, pre_label)[i]))
        print(information)


if __name__ == "__main__":
    col_dir = 'chars74k-lite/*/*.jpg'
    label,image = get_image(col_dir)
    train_images,test_images,train_labels,test_labels = split_train_test(label,image)
    train_images = data_processing(train_images)
    #test1_images = test_images[::120]
    #test1_labels = test_labels[::120]
    feature_method = "HOG"
    classification_method = "SVM"
    model,pp,pca = training(train_images,train_labels,feature_method,classification_method)
    predict_array, probs_array = test(model,pp,pca,test_images, test_labels,feature_method, classification_method)
    #prection_of_each_letter_in_test(test_labels, predict_array, classification_method)
    #test(model,pp,pca,test1_images, test1_labels,feature_method, classification_method)
