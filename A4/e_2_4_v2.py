import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import pickle
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def sigmoid(a):
	return 1/(1 + np.exp(-a))

def sigmoid_derrivative(a):
	return sigmoid(a)*(1-sigmoid(a))


def predict(X, theta1, theta2):
	(n, m) = X.shape
	a1 = np.vstack((np.ones((1,m)),X))
	z2 = np.dot(theta1.T, a1)
	a2 = np.vstack((np.ones((1,m)),sigmoid(z2)))
	a3 = sigmoid(np.dot(theta2.T, a2))
	return a3

def initialize(L_in, L_out):
	epsilon = 0.25
	return np.random.random((L_in+1, L_out))*2*epsilon - epsilon
	# return np.random.random((L_in+1, L_out))/np.sqrt(L_in+1)	#Xavier Initialization CS231n lec 5

class Network(object):
	def __init__(self,sizes):
		self.num_layers = len(sizes) - 1
		self.sizes = sizes
		self.weights = []
		for i in range(len(sizes) - 1):
			self.weights.append(initialize(sizes[i], sizes[i+1]))

	# gradient descent
	def fit(self,X, y , regconst, num_iter, alpha):	#slowest optimizer among others (#CS231n)
		cost = np.zeros(num_iter)
		for i in range(0, num_iter):
			theta1 = self.weights[0]
			theta2 = self.weights[1]

			[theta1Grad, theta2Grad, cost[i]] = self.back_propogation(X, y, regconst)
			theta1 = theta1 - alpha*theta1Grad
			theta2 = theta2 - alpha*theta2Grad
			self.weights[0] = theta1
			self.weights[1] = theta2
			# printing the current status
			if (i+1)%(num_iter*0.1) == 0:
				per = float(i+1)/num_iter*100
				print(str(per)+"% Completed, Cost:"+str(cost[i]))
		return [theta1, theta2, cost]

	#	getting paramater's gradient
	def back_propogation(self,X, y, regconst):
			# Feedforwarding to get all the layers ofr the current parameter
			theta1 = self.weights[0]
			theta2 = self.weights[1]
			(n, m) = X.shape

			z_array  = []
			a_array = []
			v_array  = []
			v = np.vstack((np.ones((1,m)),X))
			z = np.dot(theta1.T, v)
			a = z
			z_array.append(z)
			a_array.append(a)
			v_array.append(v)

			for i in range(self.num_layers-1):
				v = np.vstack((np.ones((1,m)),sigmoid(a)))
				z = np.dot(theta2.T, v)
				a = sigmoid(z)

				z_array.append(z)
				a_array.append(a)
				v_array.append(v)

			# remove !!!
			a3 = a_array[1]
			a2 = v_array[1]
			z2 = z_array[0]
			a1 = v_array[0]

			theta1F = theta1[1:,:]
			theta2F = theta2[1:,:]

			# Logistic Regression Cost function
			cost = np.sum(np.sum( -y*np.log(a3) - (1-y)*np.log(1-a3) ))/m + (regconst/(2*m))*(sum(sum(theta1F*theta1F))+sum(sum(theta2F*theta2F)))

			# Backpropagating and calculating gradient
			d3 = (a - y)*sigmoid_derrivative(z)
			d2 = (theta2F.dot(d3))*sigmoid_derrivative(z2)
			tri2 = d3.dot(a2.T)
			tri1 = d2.dot(a1.T)
			theta2Grad = tri2.T/m + (regconst/m)*np.vstack((np.zeros((1,l2)),theta2F))
			theta1Grad = tri1.T/m + (regconst/m)*np.vstack((np.zeros((1,l1)),theta1F))

			return [theta1Grad, theta2Grad, cost]



l1 = 3		# number of neurons in hidden layer
l2 = 1		# number of neurons in outer layer

X_train = np.array([[0,0],[1,0],[0,1],[1,1] ]).T
y_train = np.array([[0,1,1,0]])

X_test = X_train
y_test = y_train

(n, m) = X_train.shape
print(X_train.shape)
print(y_train.shape)

'''one_hot_y_train = np.zeros((l2,m))	# l2 * m

# One hot convertion of y
for i in range(0,m):
	one_hot_y_train[y_train[0,i], i]= 1'''

# architecture [input - hidden 1 - sigmoid - output  - sigmoid ]
eta = 0.01		# learning rate
regconst = 1	#regularization constant 	(OPTIMIZED)
num_iter = 6000	# number of times gradient descent to be return

net = Network([n,l1,l2])

[theta1, theta2, cost] = net.fit(X_train, y_train , regconst, num_iter, eta)

a3 = predict(X_test, theta1, theta2)
m = X_test.shape[1]
print(a3)
h = np.empty((1,m))
for i in range(0,m):
	h[0,i] = np.argmax(a3[:,i])

print(h)



acc = np.mean((h==y_test)*np.ones(h.shape))*100
print("With Alpha:"+str(eta)+", Regularization Const:"+str(regconst)+", Num Iteration:"+str(num_iter))
print("Accuracy:"+str(acc))

plt.plot(cost)
plt.ylabel('cost')
plt.xlabel('iteration')
plt.show()
