
import numpy as np
import csv
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file

def OLS(X,y):
    X_t = np.transpose(X)
    w = np.dot(X_t,y)
    w = np.dot(np.linalg.pinv(np.dot(X_t,X)),w)
    return w

def error_mse(X,w,y):
    N = X.shape[1] # number of data_points
    error = (np.linalg.norm(np.dot(X,w)-y))/float(N)
    return error


def plot_result(test_data,training_data,w):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Training data")
    plt.scatter(create_x(training_data),create_y(training_data),color='red')
    #plt.scatter(create_x(test_data),create_y(test_data),color='blue')
    plt.ylim(0,1.0)
    plt.xlim(0,1.0)
    x = create_x(training_data)
    plt.plot(x,w[0]+x*w[1],color='yellow',linewidth=2.0)

    plt.subplot(212)
    plt.title("Test data")
    #plt.scatter(create_x(training_data),create_y(training_data),color='red')
    plt.scatter(create_x(test_data),create_y(test_data),color='blue')
    plt.ylim(0,1.0)
    plt.xlim(0,1.0)
    x = create_x(test_data)
    plt.plot(x,w[0]+x*w[1],color='yellow',linewidth=2.0)

    plt.show()

# training model
file_name = "../regression/train_1d_reg_data.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);
w = OLS(X,y)
print('Error training',error_mse(X,w,y))

file_name = "../regression/test_1d_reg_data.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);
print('Error testing',error_mse(X,w,y))
print(w)

plot_result(test_data,training_data,w)
