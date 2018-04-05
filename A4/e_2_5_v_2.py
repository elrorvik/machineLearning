import numpy as np
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class Network(object):
    def __init__(self,sizes):
        # sizes = # neurons layer 1, # neurons layer 2, etc.

        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
        for weight in self.weights:
            print("weigh",weight.shape)



    def feedforward(self,a):
        for b,w in zip(self.biases,self.weights):
            a = sigmoid(np.dot(w,a) + b)
        return a

    def SGD (self, X_train, Y_train, epochs,eta):

        for j in range(epochs):

            # update w
            delta_b = [np.zeros(b.shape) for b in self.biases]
            delta_w = [np.zeros(w.shape) for w in self.weights]

            for i in range(len(X_train)):
                delta_nabla_b, delta_nabla_w = self.backprop(X_train[i], Y_train[i])
                delta_b = [nb+dnb for nb, dnb in zip(delta_b, delta_nabla_b)]
                delta_w = [nw+dnw for nw, dnw in zip(delta_w, delta_nabla_w)]

            self.weights  = [w-(eta)*nw for w, nw in zip(self.weights, nabla_w)]
            self.biases = [b-(eta/len(mini_batch))*nb for b, nb in zip(self.biases, nabla_b)]

            if(j%1000==0):
                print("HURRA")

    def backprop(self,x,y_t):

        #feedforward
        # initialize with input layer for activation
        activation = x.T
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
        print("delta", y_t.shape)


        delta_b[-1] = delta
        delta_w[-1] = np.dot(delta,activation_array[-2].T)

        for l in range(2,self.num_layers):
            z = z_array[-l]
            delta = np.dot(self.weights[-l+1].T,delta)*sigmoid_derrivative(z)

            delta_b[-l] = delta
            delta_w[-l] = np.dot(delta,activation_array[-l-1].T)
        return ( delta_b, delta_w)

    def evaluate(self, x, y):
        test_result = []
        for i in range(len(x)):
            test_results[i] = (np.argmax(self.feedforward(x[i])), y[i])
        return sum(int(x == y) for (x, y) in test_results)


def sigmoid_derrivative(z):
    return sigmoid(z)*(1-sigmoid(z))

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def one_hot_encode_labels(data):
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(data)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    return onehot_encoder.fit_transform(integer_encoded)


digits = load_digits()

X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
X_train, X_test = X_train/255.0, X_test/255.0
print("Y" , y_train.shape)
print("X" , X_train.shape)

one_hot_Y_train = one_hot_encode_labels(y_train)
one_hot_Y_test = one_hot_encode_labels(y_test)

temp_x = []
temp_y = []

print("Y" , one_hot_Y_test.shape)
print("X" , X_train.shape)

for i in range(len(X_train)):
    temp_x.append([X_train[i]])
    temp_y.append([one_hot_Y_train[i]])
X_train = np.array(temp_x)
one_hot_Y_train = np.array(temp_y)
print("one" , one_hot_Y_test.shape)
print("X" , X_train.shape)


exit(0)

#x = np.array([ [0,0],[1,0],[0,1],[1,1] ])
#y = np.array([[0,1,1,0]]).T
#784,30,10
print(X_train.shape)
print(one_hot_Y_train.shape)


net = Network([64,30,10])
epochs = 60
eta = 3.0
net.SGD(X_train, one_hot_Y_train, epochs,eta)




"""test = [0,1]
print(test, " classified: ", classify(w1,w2,X_test))

test = [1,0]
print(test, " classified: ", classify(w1,w2,test))

test = [1,1]
print(test, " classified: ", classify(w1,w2,test))"""
