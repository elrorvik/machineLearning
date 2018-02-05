import csv
import numpy as np

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

def create_elliptical_X(data):
    x = create_x(data)
    squared_x = np.append(np.square(x[:,0]),np.square(x[:,1]),axis = 1)
    X = create_X(data)
    X = np.append(X,squared_x,axis = 1)
    return X


def create_y(data):
    n = data.shape[1] # number of colums
    y = data[:,(n-1)]
    return y

def create_x(data):
    n = data.shape[1] # number of colums
    x = data[:,0:n-1]
    return x
