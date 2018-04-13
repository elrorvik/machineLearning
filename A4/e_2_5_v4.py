import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


def sigmoid(z):
	return 1/(1 + np.exp(-z))

def sigmoid_derrivative(z):
	return sigmoid(z)*(1-sigmoid(z))

def initialize_weights(n_in,n_out):
	e = 0.25
	return np.random.random((n_in+1, n_out))*2*e - e

class Network(object):
	def __init__(self,sizes):
		self.num_layers = len(sizes) - 1
		self.sizes = sizes
		self.weights = []
		for i in range(len(sizes) - 1):
			self.weights.append(initialize_weights(sizes[i], sizes[i+1]))

	def predict(self,X):
		(n, m) = X.shape
		a1 = np.vstack((np.ones((1,m)),X))
		z2 = np.dot(self.weights[0].T, a1)
		a2 = np.vstack((np.ones((1,m)),sigmoid(z2)))
		a3 = sigmoid(np.dot(self.weights[1].T, a2))
		return a3

	# gradient descent
	def fit(self,X, y , ephocs, alpha):
		for i in range(0, ephocs):
			[w0_nabla, w1_nabla] = self.back_propogation(X, y)
			self.weights[0] = self.weights[0] -  alpha*w0_nabla
			self.weights[1] = self.weights[1] -  alpha*w1_nabla
			# printing the current status

	def back_propogation(self,X, y):
			(n, m) = X.shape

			#feed forward
			a1 = np.vstack((np.ones((1,m)),X))
			z2 = np.dot(self.weights[0].T, a1)
			a2 = np.vstack((np.ones((1,m)),sigmoid(z2)))
			a3 = sigmoid(np.dot(self.weights[1].T, a2))

			w0_F = self.weights[0][1:,:]  # remove bias
			w1_F = self.weights[1][1:,:]

			# backpropagation and calculation of gradient
			delta_3 = (a3 - y)*sigmoid_derrivative(a3)
			delta_2 = (w1_F.dot(delta_3))*sigmoid_derrivative(z2)
			nabla_w1 = np.dot(delta_3,a2.T)
			nabla_w0 = np.dot(delta_2,a1.T)
			nabla_w1 = nabla_w1.T/m + (1.0/m)*np.vstack((np.zeros((1,l2)),w1_F)) # add bia
			nabla_w0 = nabla_w0.T/m + (1.0/m)*np.vstack((np.zeros((1,l1)),w0_F))

			return [nabla_w0, nabla_w1]



l1 = 60		# number of neurons in hidden layer
l2 = 10		# number of neurons in outer layer

digits = load_digits()
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25)
y_train = np.array([y_train])
X_train = X_train.T
y_test = np.array([y_test])
X_test = X_test.T

print(X_train.shape)

(n, m) = X_train.shape

# one hot convertion
one_hot_y_train = np.zeros((l2,m))
for i in range(0,m):
	one_hot_y_train[y_train[0,i], i]= 1


eta = 1.2	# learning rate
ephocs = 400


net = Network([n,l1,l2])

net.fit(X_train, one_hot_y_train ,  ephocs, eta)

a3 = net.predict(X_test)

m = X_test.shape[1]
h = np.empty((1,m))
for i in range(0,m):
	h[0,i] = np.argmax(a3[:,i])



print(confusion_matrix(y_test[0], h[0]))

acc = np.mean((h==y_test)*np.ones(h.shape))*100
print("With Alpha:"+str(eta)+", Num Iteration:"+str(ephocs))
print("Accuracy:"+str(acc))
