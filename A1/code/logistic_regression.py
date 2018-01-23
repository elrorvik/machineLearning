import numpy as np
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file

def sigma(w,X): # equals y_hat = estimate of y
    z = np.dot(np.transpose(w),X).item(0)
    y_hat = 1/(1+np.exp(-z))
    return y_hat

def update_w(w,X,y,eta):
    temp = 0
    for i in range(0,len(y)):
        X_i = np.transpose(X[i,:])
        y_i = y.item(i)
        temp += (sigma(w,X_i) - y_i)*X_i
    w = w - eta*temp
    return w

def cross_entropy_error(y,w,X):

    N = y.shape[0];
    temp = 0
    for i in range(0,len(y)):
        y_i = y.item(i)
        X_i = np.transpose(X[i,:])
        y_hat = sigma(w,X_i)
        #if(y_hat >= 0.5):
        #    print("Real label: ",y_i, " Guessed label: 1.0 ")
        #else:
        #    print("Real label: ",y_i, " Guessed label: 0.0 ")
        temp += y_i*np.log2(y_hat) + (1-y_i)*np.log2(1 - y_hat)
    error = -1/N*temp
    return error

def abs_delta_cross_entropy_error(y,w,X):
    delta_error = 0
    for i in range(0,len(y)):
        y_i = y.item(i)
        X_i = np.transpose(X[i,:])
        y_hat = sigma(w,X_i)
        delta_error += (y_hat - y_i)*X_i

    abs_delta_error = 0
    for i in range(0,delta_error.shape[0]):
        abs_delta_error += delta_error[i,0]*delta_error[i,0]
    return abs_delta_error


def property_plot(data,subplot_Num,fig_title,w):
    plt.figure(1)
    plt.subplot(subplot_Num)
    red_data = []
    blue_data = []
    desicion_boundary = []
    x = create_x(data);
    for i in range(0,len(data)):
        x_i = data[i,0:data.shape[1]-1].tolist()[0]
        y_i = data[i,data.shape[1]-1].tolist()
        if ( y_i == 0):
            red_data.append(x_i)
        else:
            blue_data.append(x_i)

    red_data = np.matrix(red_data)
    blue_data = np.matrix(blue_data)
    plt.scatter(red_data[:,0],red_data[:,1],color='red',label='Label=0')
    plt.scatter(blue_data[:,0],blue_data[:,1],color='blue',label='Label=1')
    x_1 =x[:,0]
    plt.plot(x_1,-w[0]/w[2] - x_1*w[1]/w[2])

    plt.legend()
    plt.title(fig_title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")


def plot_cross_entropy_error(it,error,subplot_Num):
    plt.figure(1)
    plt.subplot(subplot_Num)
    plt.plot(error)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cross-entropy error")


file_name = "../classification/cl_train_1.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);

eta = 0.1 # learning rate #0.001
w  = 0.1*np.matrix([[1],[1],[1]])
error_array = [];
it_array = [];
prev_delta_error = 100000
for i in range(0,1000):
    w = update_w(w,X,y,eta)
    error = cross_entropy_error(y,w,X)
    error_array.append(error);
    it_array.append(i);
    delta_error = abs_delta_cross_entropy_error(y,w,X)
    if(delta_error/prev_delta_error >= 0.999 ):
        break;
    else:
        #print(i, delta_error/prev_delta_error)
        #print("prev delta error: ", prev_delta_error, " delta error: ",delta_error)
        prev_delta_error = delta_error

print("The weights after training:")
print(w)
print("Error from train set",cross_entropy_error(y,w,X))

file_name = "../classification/cl_test_1.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);

print("Error from test set",cross_entropy_error(y,w,X))

property_plot(training_data,311,"Plot of trainig set",w)
property_plot(test_data,312, "Plot of test set",w)
plot_cross_entropy_error(it_array,error_array,313)
plt.show()
