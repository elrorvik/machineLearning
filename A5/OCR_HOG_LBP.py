from skimage.io import imread_collection, imshow, concatenate_images
from skimage import io
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

from OCR import label_to_letter, letter_to_label, get_image, split_train_test, data_processing, hog_feature_extraction

#https://towardsdatascience.com/scanned-digits-recognition-using-k-nearest-neighbor-k-nn-d1a1528f0dea


def local_binary_pattern_feature_extraction(image):
    lbp = local_binary_pattern(image, 24, 4, 'uniform')
    n_bins = int(lbp.max() + 1)
    df, _ = np.histogram(lbp, normed=True, bins=n_bins, range=(0, n_bins))
    return df


def train(images,labels):

    df_1_list = []
    df_2_list = []
    for i in range(len(images)):
        image = images[i]
        df_1 = hog_feature_extraction(image)
        df_2 = local_binary_pattern_feature_extraction(image)
        df_1_list.append(df_1)
        df_2_list.append(df_2)

    df_1 = np.array(df_1_list,dtype = 'float64')
    df_2 = np.array(df_2_list,dtype = 'float64')

    pp_1 = preprocessing.StandardScaler().fit(df_1)
    pp_2 = preprocessing.StandardScaler().fit(df_2)
    label_list = []
    feature_list = []
    for i in range(len(images)):
        label = labels[i][0]
        image = images[i]
        df_1 = hog_feature_extraction(image)
        df_2 = local_binary_pattern_feature_extraction(image)
        df_1 = pp_1.transform(np.array([df_1],'float64'))
        df_2  = pp_2.transform(np.array([df_2],'float64'))
        df = np.append(df_1,df_2)
        label_list.append(label)
        feature_list.append(df)

    features = np.array(feature_list,dtype = 'float64')

    model = ssv.SVC(kernel='rbf')
    model.fit(features, label_list)

    return model,pp_1,pp_2

def test(images,labels,pipe,pp_1,pp_2):
    feature_list = []
    label_list = []

    for i in range(len(images)):
        label = labels[i][0]
        image = images[i]
        df_1 = hog_feature_extraction(image)
        df_2 = local_binary_pattern_feature_extraction(image)
        df_1 = pp_1.transform(np.array([df_1],'float64'))
        df_2  = pp_2.transform(np.array([df_2],'float64'))
        df = np.append(df_1,df_2)
        label_list.append(label)
        feature_list.append(df)

    predict_array = []
    for i in range(len(feature_list)):
        feature = feature_list[i].reshape(1,-1)

        prediction = pipe.predict(feature)
        predict_array.append(prediction)

    correct = 0
    for i in range(len(predict_array)):
        if (predict_array[i] == label_list[i]):
            correct +=1
    print(correct/len(predict_array))


if __name__ == "__main__":
    col_dir = 'chars74k-lite/*/*.jpg'
    label,image = get_image(col_dir)

    train_images,test_images,train_labels,test_labels = split_train_test(label,image)
    train_images = data_processing(train_images)
    test_images = data_processing(test_images)

    pipe, pp_1,pp_2 = train(train_images,train_labels)
    test(test_images,test_labels,pipe,pp_1,pp_2)

    '''train_images = data_processing(train_images)
    test_images = data_processing(test_images)
    X = color.rgb2gray(train_images)
    y = train_labels'''

    '''label_list = []
    y_validation = []
    for i in range(train_images.shape[0]):
        train_images[i] = color.rgb2grey(train_images[i])
        train_images[i] = ndimage.median_filter(train_images[i], 3)
        train_images[i] = preprocessing.scale(train_images[i])
        label = train_labels[i][0]
        label_list.append(label);

    print(train_images.shape)

    for i in range(test_images.shape[0]):
        test_images[i] = color.rgb2grey(test_images[i])
        test_images[i] = ndimage.median_filter(test_images[i], 3)
        test_images[i] = preprocessing.scale(test_images[i])

        label = test_labels[i][0]
        y_validation.append(label);

    #X_std = StandardScaler().fit_transform(train_images)
    #X_std_validation = StandardScaler().fit_transform(test_images)
    X_std = train_images
    X_std_validation = test_images

    classifier = svm.SVC(gamma=0.005)
    classifier.fit(X_std, label_list)

    predicted = classifier.predict(X_std_validation)

    cm = confusion_matrix(y_validation,predicted)
    total = cm.sum(axis=None)
    correct = cm.diagonal().sum()

    print( "5-Component PCA Accuracy: %0.2f %%"%(100.0*correct/total))'''


    '''pca2 = PCA(n_components=20)
    pca2.fit(X_std)
    X_red = pca2.transform(X_std)
    label_list = []
    for i in range(len(train_labels)):
        label = train_labels[i][0]
        label_list.append(label);
    y_validation = []
    for i in range(len(test_labels)):
        label = test_labels[i][0]
        y_validation.append(label);

    linclass2 = KNeighborsClassifier(n_neighbors=50)
    linclass2.fit(X_red,label_list)

    X_red_validation = pca2.transform(X_std_validation)
    yhat_validation = linclass2.predict(X_red_validation)

    pca2_cm = confusion_matrix(y_validation,yhat_validation)
    total = pca2_cm.sum(axis=None)
    correct = pca2_cm.diagonal().sum()

    print( "5-Component PCA Accuracy: %0.2f %%"%(100.0*correct/total))
'''
    '''exit()
    model,pp,pca = training(train_images, train_labels,"PCA","SVM")
    print(pca)

    test(model,pp,test_images, test_labels,pca)

    exit()'''
