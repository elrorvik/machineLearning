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

def eucledian_dist_list(X,Y,row_c,k, regression = True):
    eucledian_dic = {}
    eucledian_list = [];
    x_c = X[row_c,:]
    for i in range(0,X.shape[0]):
        if(i == row_c):
            continue;
        x = X[i,:]
        dist = eucledian_dist(x,x_c)
        
        if( dist not in eucledian_dic ):
            eucledian_dic[dist] = [i]
        else:
            eucledian_dic[dist].append(i)
            
        eucledian_list.append(dist)

    eucledian_list.sort()
    
    if(regression == True):
        mean = simple_mean_k_neighbours(eucledian_list[0:k],eucledian_dic,X)
    else:
        mean = np.zeros(X.shape[1]+1)
        mean[0:X.shape[1]] = simple_mean_k_neighbours(eucledian_list[0:k],eucledian_dic,X)
        mean[-1] = classification(eucledian_list[0:k],eucledian_dic,X,Y)
        

    print(mean)

def simple_mean_k_neighbours(dist_list,dist_dic,X):
    #x_mean = x_c
    x_mean = np.zeros(X[dist_dic[dist_list[0]][0]].shape)
    last_dist = -1;
    for dist in dist_list:

        if(last_dist == dist):
            print( "i was last")
            continue;
        
        for i in dist_dic[dist]:
            x = X[i]
            x_mean += x
            
        last_dist = dist

    return x_mean/(len(dist_list))

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
    print(label_dic)

    return max_label;
     

    

def eucledian_dist(node_1,node_2):
    return np.sqrt(np.sum((node_1-node_2)**2))


filepath="dataset/knn_regression.csv"
filepath="dataset/knn_classification.csv"
X,Y = get_data(filepath)
eucledian_dist_list(X,Y,68,10,False)


