import numpy as np
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file, create_circular_X
from math import pi

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
        temp += y_i*np.log(y_hat) + (1-y_i)*np.log(1 - y_hat)
    error = -1*temp/N
    return error


def plot_dataset_in_label_color(data):
    label_0 = []
    label_1 = []

    for i in range(0,len(data)):
        x_i = data[i,0:data.shape[1]-1].tolist()[0]
        y_i = data[i,data.shape[1]-1].tolist()
        if ( y_i == 0):
            label_0.append(x_i)
        else:
            label_1.append(x_i)

    label_0 = np.matrix(label_0)
    label_1 = np.matrix(label_1)
    plt.scatter(label_0[:,0],label_0[:,1],color='red',label='Label=0')
    plt.scatter(label_1[:,0],label_1[:,1],color='blue',label='Label=1')


def property_plot_linear_separable(data,subplot_Num,fig_title,w):
    plt.figure(1)
    plt.subplot(subplot_Num)
    plot_dataset_in_label_color(data)
    x = create_x(data)
    x_1 =x[:,0]
    plt.plot(x_1,-w[0]/w[2] - x_1*w[1]/w[2], label= ' Decision boundary' ,color ='yellow')
    plt.legend()
    plt.title(fig_title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

def property_plot_circular_separable(data,subplot_Num,fig_title,w):
    plt.figure(2)
    plt.subplot(subplot_Num)
    plot_dataset_in_label_color(data)

    c = np.power(w[2,0],2)/w[4,0]/4 + np.power(w[1,0],2)/w[3,0]/4 - w[0,0] # constant used to simplify equations
    r_1 = np.sqrt(c/w[3,0])
    r_2 = np.sqrt(c/w[4,0])
    x_10 = -(w[1,0]/w[3,0]/2.0)
    x_20 = -(w[2,0]/w[4,0]/2.0)
    t = np.linspace(0, 2*pi, 100)
    plt.plot( x_10 + r_1 *np.cos(t) , x_20+r_2*np.sin(t),color='yellow',label=' Decision boundary' )
    plt.legend()
    plt.title(fig_title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")


def plot_cross_entropy_error(it,error,label):
    plt.figure(3)
    plt.plot(error,label = label)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cross-entropy error")
    plt.legend()

def test_linear_logistic_regression():
    print("Assignment 2.1")

    file_name = "../classification/cl_train_1.csv"
    training_data = get_data_from_file(file_name)
    X_train = create_X(training_data);
    y_train = create_y(training_data);

    file_name = "../classification/cl_test_1.csv"
    test_data = get_data_from_file(file_name)
    X_test = create_X(test_data);
    y_test = create_y(test_data);

    eta = 0.1 # learning rate #0.001
    w  = 0.1*np.matrix([[1],[1],[1]])
    error_train_array = [];
    error_test_array = [];
    error_train  = 0;
    error_test = 0;
    it_array = [];
    for i in range(0,1000):
        w = update_w(w,X_train,y_train,eta)

        error_train = cross_entropy_error(y_train,w,X_train)
        error_train_array.append(error_train);

        error_test = cross_entropy_error(y_test,w,X_test)
        error_test_array.append(error_test);

        it_array.append(i);

    print("The weights after training:")
    print(w)
    print("Error from train set",error_train)
    print("Error from test set",error_test)

    property_plot_linear_separable(training_data,211,"Plot of trainig set",w)
    property_plot_linear_separable(test_data,212, "Plot of test set",w)
    plot_cross_entropy_error(it_array,error_train_array,'train data')
    plot_cross_entropy_error(it_array,error_test_array,'test data')
    plt.show()


def test_circular_logistic_regression():
    file_name = "../classification/cl_train_2.csv"
    training_data = get_data_from_file(file_name)
    X = create_circular_X(training_data)
    y = create_y(training_data);

    eta = 0.1 # learning rate #0.001
    w  = 0.1*np.matrix([[1],[1],[1],[1],[1]])
    error_array = [];
    it_array = [];
    prev_delta_error = 100000
    for i in range(0,1000):
        w = update_w(w,X,y,eta)
        error = cross_entropy_error(y,w,X)
        error_array.append(error);
        it_array.append(i);
        delta_error = abs_delta_cross_entropy_error(y,w,X)
        #if(delta_error/prev_delta_error >= 0.999 ):
        #    break;
        #else:
        #    prev_delta_error = delta_error

    print("number of oterations",len(error_array))
    print("The weights after training:")
    print(w)
    print("Error from train set",cross_entropy_error(y,w,X))

    file_name = "../classification/cl_test_2.csv"
    test_data = get_data_from_file(file_name)
    X = create_circular_X(test_data)
    y = create_y(test_data);

    print("Error from test set",cross_entropy_error(y,w,X))

    property_plot_circular_separable(training_data,211,"Plot of trainig set",w)
    property_plot_circular_separable(test_data,212, "Plot of test set",w)
    plot_cross_entropy_error(it_array,error_array)
    plt.show()

#test_circular_logistic_regression
test_linear_logistic_regression()
