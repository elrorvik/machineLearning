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
from sklearn.neighbors import KNeighborsClassifier



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
    '''num = [0]*26
    print(y_train[0][0])
    for i in range(len(y_train)):
        y = int(y_train[i][0])
        num[y] += 1
    print(num)
    num = [0]*26
    for i in range(len(y_test)):
        y = int(y_test[i][0])
        num[y] += 1
    print(num)'''
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
        if(feature_method == "HOG"):
            df = hog_feature_extraction(image)
        elif(feature_method == "LBP"):
            df = local_binary_pattern_feature_extraction(image)

        feature_list.append(df)
        label_list.append(label)

    features = np.array(feature_list,dtype='float')

    pp = preprocessing.StandardScaler().fit(features)
    features = pp.transform(features)

    if(classification_method == "KNN"):
        model = KNeighborsClassifier(n_neighbors=8)
    elif(classification_method == "SVM"):
        model = ssv.SVC(kernel='rbf')

    model.fit(features,label_list)
    return model,pp

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
        label = labels[i][0]
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

def test(model,pp,images, labels):
    feature_list = []
    label_list = []

    for i in range(len(images)):
        image = images[i]
        label = labels[i][0]

        df = hog_feature_extraction(image)
        #df = local_binary_pattern_feature_extraction(image)
        df = pp.transform(np.array([df],'float64'))

        feature_list.append(df)
        label_list.append(label)

    predict_array = []
    for i in range(len(label_list)):
        predict = model.predict(feature_list[i].reshape((1,-1)))
        predict_array.append(predict)

    correct = 0
    for i in range(len(predict_array)):
        if (predict_array[i] == label_list[i]):
            correct +=1
    print(correct/len(predict_array))



col_dir = 'chars74k-lite/*/*.jpg'
label,image = get_image(col_dir)

train_images,test_images,train_labels,test_labels = split_train_test(label,image)

train_images = data_processing(train_images)
test_images = data_processing(test_images)


model,pp = training(train_images, train_labels,"HOG","SVM")
test(model,pp,test_images,test_labels)

#pipe,pp = pca_train(train_images,train_labels)
#pca_test(test_images,test_labels,pipe,pp)


#plt.figure()
#plt.imshow(train_images[4000])
#plt.figure()
#plt.imshow(train_images[10])
#print(label_to_letter(train_labels[4000]))
#print(label_to_letter(train_labels[10]))
#plt.show()
