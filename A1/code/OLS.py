
import numpy as np
import csv
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file

def linear_regression_OLS(X,y):
    X_t = np.transpose(X)
    w = np.dot(np.linalg.pinv(np.dot(X_t,X)),np.dot(X_t,y))
    return w

def error_mse(X,w,y):
    N = X.shape[0] # number of data_points
    error = np.square(np.linalg.norm(X*w-y))/N
    return error


def plot_result_linear_regression(test_data,training_data,w):
    plt.figure(1)
    plt.subplot(211)
    plt.title("Training data")
    plt.scatter(create_x(training_data),create_y(training_data),color='red')
    x = create_x(training_data)
    plt.plot(x,w[0]+x*w[1],color='yellow',linewidth=2.0)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.subplot(212)
    plt.title("Test data")
    plt.scatter(create_x(test_data),create_y(test_data),color='blue')
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    x = create_x(test_data)
    plt.plot(x,w[0]+x*w[1],color='yellow',linewidth=2.0)
    plt.show()

# Running assignemnt 1.2
print('Assignemnt 1.2: Data set 2d')
file_name = "../regression/train_2d_reg_data.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);
w = linear_regression_OLS(X,y)

print('Error training',error_mse(X,w,y))

file_name = "../regression/test_2d_reg_data.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);
print('Error testing',error_mse(X,w,y))
print("Weights after training")
print(w)

# Running assignment 1.3
print('Assignemnt 1.3: Data set 1d')
file_name = "../regression/train_1d_reg_data.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);
w = linear_regression_OLS(X,y)

print('Error training',error_mse(X,w,y))

file_name = "../regression/test_1d_reg_data.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);
print('Error testing',error_mse(X,w,y))
print("Weights after training")
print(w)

plot_result_linear_regression(test_data,training_data,w)
