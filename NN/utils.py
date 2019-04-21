import numpy as np
def getMNIST(datapath):
    x_train = list(np.load(datapath +"x_train.npy"))
    x_test = list(np.load(datapath +"x_test.npy"))
    y_train = list(np.load(datapath +"y_train.npy"))
    y_test = list(np.load(datapath +"y_test.npy"))
    return x_train,x_test,y_train,y_test

