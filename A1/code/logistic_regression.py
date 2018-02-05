import numpy as np
import matplotlib.pyplot as plt
from utility import create_X, create_x ,create_y,get_data_from_file, create_elliptical_X
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


def property_plot_linear_separable(data,subplot_Num,fig_title,w,fig_num):
    plt.figure(fig_num)
    plt.subplot(subplot_Num)
    plot_dataset_in_label_color(data)
    x = create_x(data)
    x_1 =x[:,0]
    plt.plot(x_1,-w[0]/w[2] - x_1*w[1]/w[2], label= ' Decision boundary' ,color ='yellow')
    plt.legend()
    plt.title(fig_title)
    plt.xlabel("$x_1$")
    plt.ylabel("$x_2$")

def property_plot_elliptical_separable(data,subplot_Num,fig_title,w,fig_num):
    plt.figure(fig_num)
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


def plot_cross_entropy_error(error,label,fig_num):
    plt.figure(fig_num)
    plt.plot(error,label = label)
    plt.xlabel("Number of iterations")
    plt.ylabel("Cross-entropy error")
    plt.legend()


def test_linear_logistic_regression(file_name_train,file_name_test,fig_num):
    training_data = get_data_from_file(file_name_train)
    X_train = create_X(training_data); #input X = [1, x_1,x_2]
    y_train = create_y(training_data);

    test_data = get_data_from_file(file_name_test)
    X_test = create_X(test_data);
    y_test = create_y(test_data);

    eta = 0.1 # learning rate #0.001
    w  = 0.1*np.matrix([[1],[1],[1]])
    train_error_array = [];
    test_error_array = [];
    train_error  = 0;
    test_error = 0;

    for i in range(0,1000):
        w = update_w(w,X_train,y_train,eta)

        train_error = cross_entropy_error(y_train,w,X_train)
        train_error_array.append(train_error);

        test_error = cross_entropy_error(y_test,w,X_test)
        test_error_array.append(test_error);

    print("The weights after training:")
    print(w)
    print("Error from train set",train_error)
    print("Error from test set",test_error)

    property_plot_linear_separable(training_data,211,"Plot of training set",w,fig_num)
    property_plot_linear_separable(test_data,212, "Plot of test set",w,fig_num)
    plot_cross_entropy_error(train_error_array,'train data',fig_num+1)
    plot_cross_entropy_error(test_error_array,'test data',fig_num + 1)



def test_elliptical_logistic_regression(file_name_train,file_name_test,fig_num):

    training_data = get_data_from_file(file_name_train)
    X_train = create_elliptical_X(training_data) #input X = [1, x_1,x_2,x_1^2,x_2^2]
    y_train = create_y(training_data);

    test_data = get_data_from_file(file_name_test)
    X_test = create_elliptical_X(test_data)
    y_test = create_y(test_data);

    eta = 0.10 # learning rate #0.001
    w  = 0.1*np.matrix([[1],[1],[1],[1],[1]])
    train_error_array = [];
    test_error_array = [];
    train_error  = 0;
    test_error = 0;
    it_array = [];

    for i in range(0,1000):
        w = update_w(w,X_train,y_train,eta)

        train_error = cross_entropy_error(y_train,w,X_train)
        train_error_array.append(train_error);

        test_error = cross_entropy_error(y_test,w,X_test)
        test_error_array.append(test_error);

        it_array.append(i);

    print("The weights after training:")
    print(w)
    print("Error from train set",train_error)
    print("Error from test set",test_error)

    property_plot_elliptical_separable(training_data,211,"Plot of trainig set",w,fig_num)
    property_plot_elliptical_separable(test_data,212, "Plot of test set",w,fig_num)
    plot_cross_entropy_error(train_error_array,'train data',fig_num+1)
    plot_cross_entropy_error(test_error_array,'test data',fig_num+1)


# assignemnt 1
print("Assignemtn 2.1")
file_name_train = "../classification/cl_train_1.csv"
file_name_test = "../classification/cl_test_1.csv"
test_linear_logistic_regression(file_name_train,file_name_test,1)

# assignemtn 2
print("\n Assignemtn 2.2")
file_name_train = "../classification/cl_train_2.csv"
file_name_test = "../classification/cl_test_2.csv"
print("Linear decision boundary")
test_elliptical_logistic_regression(file_name_train,file_name_test,5)
print("\n Elliptical decision")
test_linear_logistic_regression(file_name_train,file_name_test,3)
plt.show()
