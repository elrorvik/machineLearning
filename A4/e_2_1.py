import matplotlib.pyplot as plt
import csv
import numpy as np
import operator

def get_data(filepath):
    try:
        f = open(filepath,'r')
    except IOError:
        print("could not open file")
    
    data = np.genfromtxt(filepath,delimiter=',')

    return data[:,0:data.shape[1]-1],data[:,-1]

def eucledian_dist_list(X,Y,x_c,k, regression = True):
    eucledian_dic = {}
    eucledian_list = [];

    for i in range(0,X.shape[0]):
        x = X[i,:]
        dist = eucledian_dist(x,x_c)
        
        if( dist not in eucledian_dic ):
            eucledian_dic[dist] = [i]
        else:
            eucledian_dic[dist].append(i)
            
        eucledian_list.append(dist)

    eucledian_list.sort()

    print_k_neighbours(eucledian_list[0:k],eucledian_dic,X,Y)
    if(regression == True):
        regression = mean_regression_k_neighbours(eucledian_list[0:k],eucledian_dic,X,Y)
        print(x_c, " regression value: ", regression)
    else:
        label = classification(eucledian_list[0:k],eucledian_dic,X,Y)
        print(x_c, " classification value: ", label)
        


def print_k_neighbours(dist_list,dist_dic,X,Y):
    #x_mean = x_c
    y_mean = 0
    last_dist = -1;
    print("Neighbours")
    for dist in dist_list:
        if(last_dist == dist):
            continue;
        for i in dist_dic[dist]:
            print(X[i], " -> ", Y[i])
              
        last_dist = dist


def mean_regression_k_neighbours(dist_list,dist_dic,X,Y):
    #x_mean = x_c
    y_mean = 0
    last_dist = -1;
    for dist in dist_list:
        if(last_dist == dist):
            continue;
        for i in dist_dic[dist]:
            y = Y[i]
            y_mean += y
              
        last_dist = dist

    return y_mean/(len(dist_list))

def classification(dist_list,dist_dic,X,Y):
    #x_mean = x_c
    label_dic = {}
    last_dist = -1;
    for dist in dist_list:
        if(last_dist == dist):
            print( "i was last")
            continue;
        
        for i in dist_dic[dist]:
            y = Y[i]
            if( y not in label_dic ):
                label_dic[y] = 1
            else:
                label_dic[y] += 1

    max_label = max(label_dic.items(), key=operator.itemgetter(1))[0]
    return max_label;
     

    

def eucledian_dist(node_1,node_2):
    return np.sqrt(np.sum((node_1-node_2)**2))


filepath="dataset/knn_regression.csv"
X,Y = get_data(filepath)

x_c = np.array([6.3,2.7,4.91])
regression = True
eucledian_dist_list(X,Y,x_c,10,regression)

filepath="dataset/knn_classification.csv"
X,Y = get_data(filepath)

x_c = np.array([6.3,2.7,4.91,1.8])
regression = False
eucledian_dist_list(X,Y,x_c,10,regression)


