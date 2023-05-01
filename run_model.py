# -*- coding: utf-8 -*-
"""
Created on Fri Apr 21 18:30:22 2023

@author: gbonnet
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from FC_nn import FC


if __name__ == '__main__':
    print('test')
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
    # mnist
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    for i in range(3): 
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
        plt.show()
    print('X_train: ' + str(train_X.shape))
    print('Y_train: ' + str(train_y.shape))
    print('X_test:  '  + str(test_X.shape))
    print('Y_test:  '  + str(test_y.shape))
    
    #%%
    train_X_reshape = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
    train_y_reshape = train_y.reshape(1, train_y.shape[0])
    test_X_reshape = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T
    test_y_reshape = test_y.reshape(1, test_y.shape[0])
    print('X_train: ' + str(train_X_reshape.shape))
    print('Y_train: ' + str(train_y_reshape.shape))
    print('X_test:  '  + str(test_X_reshape.shape))
    print('Y_test:  '  + str(test_y_reshape.shape))
    
    #%%
    hidden_layers = np.array([4, 32, 32, 10])
    my_layer = FC()
    print(train_y_reshape)
    parameters = my_layer.train_nn(train_X_reshape/255, train_y_reshape, 
                                   hidden_layers,
                                   test_X_reshape/255, test_y_reshape)