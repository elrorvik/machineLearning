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
        row = [float(row[column]) for column in range(0,len(row))]
        ret.append(row)
    ret = np.matrix(ret)
    f.close()
    return ret

def create_X(data):
    m = data.shape[0] # number of rows
    x = create_x(data)
    ones = np.ones((m,1))
    X = np.append(ones,x,axis = 1)
    return X

def create_y(data):
    n = data.shape[1] # number of colums
    y = data[:,(n-1)]
    return y

def create_x(data):
    n = data.shape[1] # number of colums
    x = data[:,0:n-1]
    return x

def sigma(w,x):
    z = np.dot(np.transpose(w),x)
    print(z)
    return 1/(1+np.exp(-z))

def update_w(w,X,y,eta):
    temp = 0
    for i in range(0,len(y)):
        x_i = np.transpose(X[i,:])
        y_i = y[i]
        print(sigma(w,x_i)- y_i)
        temp += (sigma(w,x_i)- y_i)*x_i
    w = w - eta*temp
    return w

def predict(x,w):
    y = 0
    if np.dot(np.transpose(w),x) >= 0 :
        y = 1

    return y

def plot_result(training_data,w):
    plt.figure(1)
    #plt.subplot(211)
    plt.title("Training data")
    plt.scatter(create_x(training_data),create_y(training_data),color='red')
    #plt.scatter(create_x(test_data),create_y(test_data),color='blue')
    #plt.ylim(0,1.0)
    #plt.xlim(0,1.0)
    #x = create_x(training_data)
    #plt.plot(x,w[0]+x*w[1],color='yellow',linewidth=2.0)


    plt.show()

file_name = "../classification/cl_train_1.csv"
training_data = get_data_from_file(file_name)
X = create_X(training_data);
y = create_y(training_data);

eta = 0.1 # learning rate
w  = np.ones(X.shape[1])
for i in range(0,500):
    w = update_w(w,X,y,eta)

file_name = "../classification/cl_test_1.csv"
