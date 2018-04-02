import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z): 
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

        z2_delta = -(y - z2)*(z2*(1-z2))
        z1_delta = z2_delta.dot(w2.T) * (z1 * (1-z1))

        delta_loss_w2 = z1.T.dot(z2_delta)
        delta_loss_w1 = z0.T.dot(z1_delta)
        
        w2 -= alpha*delta_loss_w2
        w1 -= alpha*delta_loss_w1
    return w1,w2

x = np.array([ [0,0],[1,0],[0,1],[1,1] ])
y = np.array([[0,1,1,0]]).T
w1 = 2*np.random.random((3,4)) - 1
w2 = 2*np.random.random((4,1)) - 1
w1,w2 = train(x,y,w1,w2)

test = [0,0]
print(test, " classified: ", classify(w1,w2,test))

test = [0,1]
print(test, " classified: ", classify(w1,w2,test))

test = [1,0]
print(test, " classified: ", classify(w1,w2,test))

test = [1,1]
print(test, " classified: ", classify(w1,w2,test))



