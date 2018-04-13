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

def accuracy(w1,w2):
    loss = 0
    test = [0,0]
    loss += np.square(( 0 - classify(w1,w2,test)))
    test = [0,1]
    loss += np.square(( 1 - classify(w1,w2,test)))
    test = [1,0]
    loss += np.square(( 1 - classify(w1,w2,test)))
    test = [1,1]
    loss += np.square(( 0 - classify(w1,w2,test)))
    return loss/2.0

def train(x,y_t,w1,w2):
    X=np.hstack((np.array([1]*x.shape[0]).reshape(x.shape[0],1),x))
    epochs = 2000
    alpha = 1 #learning rate
    accuracy_array  = []
    for j in range(epochs):
        #forward
        z0 = X
        z1 = sigmoid(np.dot(z0,w1)) # input layer
        z2 = sigmoid(np.dot(z1,w2)) # hidden 1
    #Backtracking
        z2_delta = -(y - z2)*(z2*(1-z2))
        z1_delta = z2_delta.dot(w2.T) * (z1 * (1-z1))

        delta_loss_w2 = z1.T.dot(z2_delta)
        delta_loss_w1 = z0.T.dot(z1_delta)

        w2 -= alpha*delta_loss_w2
        w1 -= alpha*delta_loss_w1
        accuracy_array.append(accuracy(w1,w2))
    return w1,w2,accuracy_array

x = np.array([ [0,0],[1,0],[0,1],[1,1] ]) #training input
y = np.array([[0,1,1,0]]).T #training labels
w1 = 2*np.random.random((3,4)) - 1 #init weights input layer
w2 = 2*np.random.random((4,1)) - 1 #init weights hidden layer
w1,w2,accuracy_array = train(x,y,w1,w2)

test = [0,0]
print(test, " classified: ", classify(w1,w2,test))

test = [0,1]
print(test, " classified: ", classify(w1,w2,test))

test = [1,0]
print(test, " classified: ", classify(w1,w2,test))

test = [1,1]
print(test, " classified: ", classify(w1,w2,test))

plt.plot(accuracy_array)
plt.ylabel("Accuracy")
plt.xlabel("ephocs")
plt.show()
