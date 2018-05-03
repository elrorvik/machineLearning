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

        #lineImgArray = sciImg.reshape((1,400)) #why ?????
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

    for i in range(len(images)):
        image = images[i]
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
    for i in range(len(label_list)):
        predict = model.predict(feature_list[i].reshape((1,-1)))
        probs = model.predict_proba(feature_list[i].reshape((1,-1)))

        predict_array.append(predict[0])
        probs_array.append(probs[0][int(predict)])

    #print(probs, predict_array[-1])
    correct = 0
    for i in range(len(predict_array)):
        if (predict_array[i] == label_list[i]):
            correct +=1

    print("accuracy" , correct/len(predict_array))
    return predict_array, probs_array

def pca_train(images,labels):
    feature_list = []
    label_list = []

    for i in range(len(images)):
        label = labels[i][0]
        image = images[i]
        df = hog_feature_extraction(image)
        #df = local_binary_pattern_feature_extraction(image)
        feature_list.append(df)
        label_list.append(label)

    hog_features = np.array(feature_list,dtype = 'float64')

    pp = preprocessing.StandardScaler().fit(hog_features)
    features = pp.transform(hog_features)

    pca = PCA(n_components=600)
    model = LogisticRegression()
    #model = ssv.SVC(kernel='rbf')

    pipe = Pipeline([('pca', pca), ('logistic', model)])
    pipe.fit(features, label_list)

    return pipe,pp

def pca_test(images,labels,pipe,pp):
    feature_list = []
    label_list = []

    for i in range(len(images)):
        label = labels[i]
        image = images[i]

        df = hog_feature_extraction(image)
        #df = local_binary_pattern_feature_extraction(image)
        df = pp.transform(np.array([df],'float64'))

        feature_list.append(df)
        label_list.append(label);

    predict_array = []
    for i in range(len(feature_list)):
        feature = feature_list[i]
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

    feature_method = "HOG+PCA"
    classification_method = "SVM"
    model,pp,pca = training(train_images,train_labels,feature_method,classification_method)
    test(model,pp,pca,test_images, test_labels,feature_method, classification_method)

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
