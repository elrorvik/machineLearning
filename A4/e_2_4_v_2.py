import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from scipy.special import expit, logit


class Network(object):
    def __init__(self,sizes):
        # sizes = # neurons layer 1, # neurons layer 2, etc.

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]



    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a)+b)
        return a

    def SGD (self, X_train, Y_train, epochs,eta,X_test,Y_test ):

        for j in range(epochs):
            #random.shuffle(training_data)
            # update w
            nabla_b = [np.zeros(b.shape) for b in self.biases]
            nabla_w = [np.zeros(w.shape) for w in self.weights]
            for i in range(len(X_train)):
                x = X_train[i]
                y = Y_train[i]
                delta_nabla_b, delta_nabla_w = self.backprop(x, y)
                nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
                nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            self.weights = [w-(eta)*nw
                            for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta)*nb
                           for b, nb in zip(self.biases, nabla_b)]
            if j % 10 == 0 :
                n_test = len(X_test)
                print( "Epoch {0}: {1} / {2}".format(j, self.evaluate(X_test,Y_test), n_test))


    def backprop(self,x,y_t):

        #feedforward
        # initialize with input layer for activation
        activation = x
        activation_array = [x]
        z_array = []

        for b,w in zip(self.biases,self.weights):
            z = np.dot(w,activation) + b
            z_array.append(z)
            activation = sigmoid(z)
            activation_array.append(activation)

        # backward pass
        delta_b = [np.zeros(b.shape) for b in self.biases]
        delta_w = [np.zeros(w.shape) for w in self.weights]

        y = activation_array[-1]
        delta = -(y_t - y)*sigmoid_derrivative(z)


        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta,activation_array[-2].T)

        for l in range(2,self.num_layers):
            z = z_array[-l]
            delta = np.dot(self.weights[-l+1].T,delta)*sigmoid_derrivative(z)

            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta,activation_array[-l-1].T)
        return ( delta_b, delta_w)

    def evaluate(self, x, y):
        test_results = [(np.argmax(self.feedforward(x[i])), y[i])
                        for i in range(len(x))]
        num_correct = 0
        for (x,y) in test_results:
            if( x == y):
                num_correct += 1
        return num_correct


def sigmoid_derrivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+expit(-z))


def one_hot_encode_labels(data):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    print(" one hot " ,len(integer_encoded))
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)

def transform_shape(data):
    temp = []
    for i in range(len(data)):
        t_temp = []
        for j in range(len(data[i])):
            t_temp.append([data[i][j]])
        temp.append(t_temp)
    return np.array(temp)



X_train = np.array([ [0,0],[1,0],[0,1],[1,1] ])
Y_train = np.array([0,1,1,0])

#X_train = transform_shape(X_train)
#Y_train = transform_shape(Y_train)


print("Y test" , Y_train.shape)
print("X test" , X_train.shape)
X_test = X_train
Y_test = Y_train


net = Network([2,2,1])
epochs = 60000
eta = 0.01
net.SGD(X_train, Y_train, epochs,eta, X_test,Y_test)





"""test = [0,1]
print(test, " classified: ", classify(w1,w2,X_test))

test = [1,0]
print(test, " classified: ", classify(w1,w2,test))

test = [1,1]
print(test, " classified: ", classify(w1,w2,test))"""
