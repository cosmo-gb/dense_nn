# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:30:22 2023

@author: gbonnet
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from sklearn.datasets import make_circles
from FC_nn import FC


if __name__ == '__main__':
    print('test')
    #%%
    ###########################################################################
    ####################### SIMPLE EXAMPLE ####################################
    ###########################################################################
    # how to recognise if a data point is inside a circle or inside another circle
    N_tot = 1000
    N_train = 500
    N_test = N_tot - N_train
    X, y = make_circles(n_samples=N_tot, noise=0.1, factor=0.3, random_state=0)
    plt.figure(figsize=(2,2))
    plt.scatter(X[:N_train,0], X[:N_train,1], 
                s=0.5, c=y[:N_train])
    plt.show()
    
    #%%
    # reorganize the data
    X_train = X.T # (n_samples, n_features) -> (n_features, n_samples)
    y_train = y.reshape(1, len(y)) # (n_samples, n_features) -> (n_features, n_samples)
    
    #%%
    # train the network
    my_layer = FC()
    my_layer.n_iter = 10000
    my_layer.stop_th = 10
    hidden_layers = np.array([32, 32, 32])
    parameters = my_layer.train_nn(X_train[:,:N_train], y_train[:,:N_train], 
                                   hidden_layers,
                                   X_train[:,-N_test:], y_train[:,-N_test:])
    
    #%%
    # test the prediction
    y_pred_test = my_layer.predict(X_train[:,-N_test:], parameters)
    plt.figure(figsize=(2,2))
    plt.scatter(X_train[0,-N_test:], X_train[1,-N_test:], 
                c=y_pred_test, s=0.5)
    plt.show()
    
    
    #%%
    # data from mchine learnia
    from utilities import load_data
    X_train, y_train, X_test, y_test = load_data()
    
    #%%
    # reorganize train and test set: dog vs cat picture
    # x: (n_samples, n_pixels, n_pixels) -> (n_pixels*n_pixels, n_samples)
    X_train_reshape = X_train.reshape(X_train.shape[0], X_train.shape[1] * X_train.shape[2]).T
    # y: (n_samples, 1) -> (1, n_samples)
    y_train_reshape = y_train.reshape(1, y_train.shape[0]).astype(int)
    X_test_reshape = X_test.reshape(X_test.shape[0], X_test.shape[1] * X_test.shape[2]).T
    y_test_reshape = y_test.reshape(1, y_test.shape[0]).astype(int)
    
    #%%
    # train model
    my_layer = FC()
    my_layer.n_iter = 3000
    my_layer.stop_th = 100
    hidden_layers = np.array([32, 32, 32, 32])
    parameters = my_layer.train_nn(X_train_reshape/X_train_reshape.max(), y_train_reshape,
                                   hidden_layers,
                                   X_test_reshape/X_train_reshape.max(), y_test_reshape)
    
    
    
    #%%
    ###########################################################################
    ####################### MNIST #############################################
    ###########################################################################
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for i in range(3): 
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
        plt.show()
    #print('X_train: ' + str(train_X.shape))
    #print('Y_train: ' + str(train_y.shape))
    #print('X_test:  '  + str(test_X.shape))
    #print('Y_test:  '  + str(test_y.shape))
    # reshape
    train_X_reshape = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
    train_y_reshape = train_y.reshape(1, train_y.shape[0])
    test_X_reshape = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T
    test_y_reshape = test_y.reshape(1, test_y.shape[0])
    #print('X_train: ' + str(train_X_reshape.shape))
    #print('Y_train: ' + str(train_y_reshape.shape))
    #print('X_test:  '  + str(test_X_reshape.shape))
    #print('Y_test:  '  + str(test_y_reshape.shape))
    
    # reorganize y data, as I need 0 or 1 on each output, 
    # and not 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
    y_train = np.zeros((10, train_y_reshape.shape[1]))
    y_test = np.zeros((10, test_y_reshape.shape[1]))
    for i in range(10):
        ind_train_i = np.where(train_y_reshape[0] == i)[0]
        y_train[i,ind_train_i] = 1
        ind_test_i = np.where(test_y_reshape[0] == i)[0]
        y_test[i,ind_test_i] = 1
    
    #%%
    hidden_layers = np.array([32, 32])
    my_layer = FC()
    my_layer.n_epochs = 3
    my_layer.stop_th = 10
    my_layer.verbose = 3
    print(y_train.shape)
    parameters, measure = my_layer.train_nn(train_X_reshape/255, y_train, 
                                   hidden_layers,
                                   test_X_reshape/255, y_test)
    print(parameters['W2'][0:2,0:2])