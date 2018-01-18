
import numpy as np
import csv
import matplotlib.pyplot as plt

def get_data_from_file(file_name):
    try:
        f = open(file_name,'r')
    except IOError:
        print("could not open file")

    reader = csv.reader(f, delimiter=",")
    ret = []
    for row in reader:
        row = [float(row[column]) for column in range(0,len(row)-1)]
        ret.append(row)

    ret = np.matrix(ret)
    f.close()
    return ret

def OLS(X,y):
    X_t = np.transpose(X)
    w = np.dot(X_t,y)
    w = np.dot(np.linalg.pinv(np.dot(X_t,X)),w)
    return w

def error_mse(X,w,y):
    error = np.linalg.norm(np.dot(X,w)-y)
    return error

def create_X(data):
    m = data.shape[0] # number of rows
    x = data[:,0]
    ones = np.ones((m,1))
    X = np.append(ones,x,axis = 1)
    return X

def create_y(data):
    n = data.shape[1] # number of colums
    y = data[:,(n-1)]
    return y

def create_x(data):
    n = data.shape[1] # number of colums
    x = data[:,n-1]
    return x

def plot_result(test_data,training_data):
    plt.scatter(create_x(training_data),create_y(training_data),color='red')
    plt.scatter(create_x(test_data),create_y(test_data),color='blue')
    plt.show()

# training model
file_name = "../regression/train_2d_reg_data.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);
w = OLS(X,y)
print('Error training',error_mse(X,w,y))

file_name = "../regression/test_2d_reg_data.csv"
test_data = get_data_from_file(file_name)
X = create_X(test_data);
y = create_y(test_data);
print('Error testing',error_mse(X,w,y))
print(w)

#plot_result(test_data,training_data)
