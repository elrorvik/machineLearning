import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder



"""print(digits.data.shape)
plt.figure()
plt.gray()
plt.matshow(digits.images[0])
plt.show()"""

def sigmoid(z):
    print(z)
    if(z >100000):
        return 1.0
    return 1.0/(1.0+np.exp(-z))

def classify(w1,w2,x):
    X = np.hstack(([1],x))
    z1 = sigmoid(np.dot(X,w1)) # layer 1
    z2 = sigmoid(np.dot(z1,w2))
    #return z2
    return 0 if (z2<0.5) else 1

def delta_loss(y_t,y,w,x):
    return

def update_w(w,alpha,delta_L):
    return w - alpha*delta_L

def train(x,y_t,w1,w2):
    X=np.hstack((np.array([1]*x.shape[0]).reshape(x.shape[0],1),x))
    epochs = 60000
    alpha = 1
    for j in range(epochs):
        z0 = X
        z1 = sigmoid(np.dot(z0,w1)) # hidden 1
        z2 = sigmoid(np.dot(z1,w2)) # hidden 2

        z2_delta = -(y_t - z2)*(z2*(1-z2))
        z1_delta = z2_delta.dot(w2.T) * (z1 * (1-z1))

        delta_loss_w2 = z1.T.dot(z2_delta)
        delta_loss_w1 = z0.T.dot(z1_delta)

        w2 -= alpha*delta_loss_w2
        w1 -= alpha*delta_loss_w1
        if(j%1000==0):
            print("HURRA")
    return w1,w2

def one_hot_encode_labels(data):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)


digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
X_train, X_test = X_train/255.0, X_test/255.0
one_hot_Y_train = one_hot_encode_labels(y_train)
one_hot_Y_test = one_hot_encode_labels(y_test)
print(X_train.shape)

#x = np.array([ [0,0],[1,0],[0,1],[1,1] ])
#y = np.array([[0,1,1,0]]).T
w1 = 2*np.random.random((X_train.shape[1] +1,X_train.shape[0]))-1
w2 = 2*np.random.random((X_train.shape[0],10)) - 1
w1,w2 = train(X_train,one_hot_Y_train,w1,w2)

test = [0,0]
print(one_hot_Y_test[0], " classified: ", classify(w1,w2,X_test[0]))

"""test = [0,1]
print(test, " classified: ", classify(w1,w2,X_test))

test = [1,0]
print(test, " classified: ", classify(w1,w2,test))

test = [1,1]
print(test, " classified: ", classify(w1,w2,test))"""
