import numpy as np
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file

def sigma(w,x):
    z = np.dot(np.transpose(w),x)
    z = z.item(0)
    return 1/(1+np.exp(-z))

def update_w(w,X,y,eta):
    temp = 0
    for i in range(0,len(y)):
        x_i = np.transpose(X[i,:])
        y_i = y.item(i)
        temp += (sigma(w,x_i) - y_i)*x_i
    w = w - eta*temp
    return w

def predict(x,w):
    y = 0
    if np.dot(np.transpose(w),x) >= 0.5 :
        y = 1
    return y

def cross_entropy_error(y,w,x):
    N = y.shape[0];
    temp = 0
    for i in range(0,len(y)):
        y_i = y.item(i)
        x_i = x.item(i)
        sigma_z = sigma(w,x_i)
        temp += y_i*np.log2(sigma_z) + (1-y_i)*np.log2(1 - sigma_z)
    error = -1/N*temp
    return error

def plot_linearity(data,subplot_Num,fig_title):
    plt.figure(1)
    plt.subplot(subplot_Num)
    red_data = []
    blue_data = []

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

eta = 0.001 # learning rate
w  = 1*np.matrix([[1,1,1],[1,1,1],[1,1,1]])
error_array = [];
it_array = [];
for i in range(0,1000):
    w = update_w(w,X,y,eta)
    error = cross_entropy_error(y,w,X)
    error_array.append(error);
    it_array.append(i);

print(w)
print("Error from train set",cross_entropy_error(y,w,X))

file_name = "../classification/cl_test_1.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);

print("Error from test set",cross_entropy_error(y,w,X))

plot_linearity(training_data,311,"Plot of trainig set")
plot_linearity(test_data,312, "Plot of test set")
plot_cross_entropy_error(it_array,error_array,313)
plt.show()
