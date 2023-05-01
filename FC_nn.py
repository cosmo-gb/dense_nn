# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:34:16 2023

@author: gbonnet
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss, accuracy_score
from sklearn.datasets import make_circles
from keras.datasets import mnist

class FC :
    
    def __init__(self):
        self.learning_rate = 0.01
        self.n_iter = 100
        self.stop_th = 10
        self.verbose = 1
        
    def initialization(self, dimensions: np.ndarray) -> dict:
        '''
        Initialize the parameters W and b of the network
        Parameters
        ----------
        dimensions : np.array, contains the the dimension of each layer,
                    dimensions[0] being the dimension of the input, 
                    e.g. the dimension of the first sample element
                    dimensions[1] being the number of unit of the first layer
                    dimensions[2] being the number of unit of the second layer, ...

        Returns
        -------
        parameters : dictionnary, key being the Wl and bl parameteres
                    and values being the values of these parameters
        '''
        parameters = {}
        
        for l in range(1, self.N_layers+1) :
            parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
            parameters['b' + str(l)] = np.random.randn(dimensions[l], 1)
        
        return parameters
    
    
    def forward_propagation(self, X, parameters) :
        
        activations = {'A0': X}
        
        for l in range(1, self.N_layers+1) :
            # Z_l has a shape (n_unit_l, n_sample)
            Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
            # be carefull, this is activation function dependent
            # activation_l has a shape (n_unit_l, n_sample)
            # => activation_l-1 has a shape (n_unit_l-1, n_sample)
            activations['A'+str(l)] = 1 / (1 + np.exp(-Z))
        
        return(activations)
    
    def back_propagation(self, y, activations, parameters):
        
        m = y.shape[1] # Number of sample
        
        # dZ of layer l has a shape of (n_unit_l, n_sample)
        # activation of layer l-1 has a shape of (n_unit_l-1, n_sample)
        # its transpose has thus a shape of (n_sample, n_unit_l-1)
        # thus the matricial product of dZ_l times activations_l-1.transpose
        # has a shape of (n_unit_l, n_unit_l-1)
        # this is the gradient shape, equal to the W shape
        # initialization: dZ_lfinal
        dZ = activations['A' + str(self.N_layers)] - y
        gradients = {}
        
        for l in reversed(range(1, self.N_layers + 1)):
            gradients['dW' + str(l)] = (1 / m) * np.dot(dZ, activations['A' + str(l-1)].T)
            gradients['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1 : # dZ0 is meaningless
                # dZ_l-1 = dot(W_l.T , dZ_l) * A_l-1 * (1 - A_l-1)
                dZ = np.dot(parameters['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] \
                    * (1 - activations['A' + str(l - 1)])
        
        return gradients
    
    def update(self, gradients, parameters):
        
        # the parameters are updated after 
        # having forward and backward all the sample
        for l in range(1, self.N_layers + 1):
            parameters['W' + str(l)] = parameters['W' + str(l)] - self.learning_rate * gradients['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - self.learning_rate * gradients['db' + str(l)]
            
        return parameters
    
    def predict(self, X, parameters):
        activations = self.forward_propagation(X, parameters)
        #A_final = np.argmax(activations['A' + str(self.N_layers)], axis=0)
        A_final = activations['A' + str(self.N_layers)]
        return A_final >= 0.5
    
    def early_stop_inf(self, f_true_test, stop):
        if f_true_test[-1] < f_true_test[-2-stop] :
            stop += 1
            print('new test:',f_true_test[-1],' < ',f_true_test[-2-stop],'old test')
        else :
            stop = 0
        return stop
    
    def early_stop_sup(self, f_true_test, stop):
        if f_true_test[-1] > f_true_test[-2-stop] :
            stop += 1
            print('new test:',f_true_test[-1],' > ',f_true_test[-2-stop],'old test')
        else :
            stop = 0
        return stop
                
    
    def train_nn(self, X, y, hidden_layers, X_test, y_test) :
        
        np.random.seed(0)
        
        dimensions = np.insert(hidden_layers, 0, X.shape[0])
        dimensions = np.append(dimensions, y.shape[0])
        self.N_layers = len(dimensions) - 1
        self.N_samples_train = y.shape[1]
        self.N_samples_test = y_test.shape[1]
        
        parameters = self.initialization(dimensions)
        
        f_true_train, f_true_test = [], []
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []
        stop = 0
        for i in range(self.n_iter):
            activations = self.forward_propagation(X, parameters)
            gradients = self.back_propagation(y, activations, parameters)
            parameters = self.update(gradients, parameters)
            if i % 1 == 0:
                train_loss.append(log_loss(y, activations['A' + str(self.N_layers)])/self.N_samples_train)
                y_pred = self.predict(X, parameters)
                train_acc.append(accuracy_score(y.flatten(), y_pred.flatten()))
                
                # test
                activations_test = self.forward_propagation(X_test, parameters)
                test_loss.append(log_loss(y_test, activations_test['A' + str(self.N_layers)])/self.N_samples_test)
                y_pred_test = self.predict(X_test, parameters)
                test_acc.append(accuracy_score(y_test.flatten(), y_pred_test.flatten()))
                if i > 1 :
                    stop = self.early_stop_inf(test_acc, stop)
                    if stop > self.stop_th :
                        break
            if i % self.verbose == 0:
                print('epoch:',i)
                print('test_loss:',test_loss[i])
                print('test_acc:',test_acc[i])
        # visualisation
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(3, 3))
        ax[0].plot(train_loss, c='b', label='loss train')
        ax[0].plot(test_loss, c='r', label='loss test')
        ax[0].legend()
        ax[1].plot(train_acc, c='b', label='acc train')
        ax[1].plot(test_acc, c='r', label='acc test')
        ax[1].legend()
        plt.show()
        
        return parameters
    

    
    
if __name__ == '__main__' :
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
    ###########################################################################
    ####################### MNIST #############################################
    ###########################################################################
    # mnist data loading
    (train_X, train_y), (test_X, test_y) = mnist.load_data()
    # show mnist data
    for i in range(3): 
        plt.subplot(330 + 1 + i)
        plt.imshow(train_X[i], cmap=plt.get_cmap('gray'))
        plt.show()
    
    #%%
    # reshape x and y mnist data
    # x: (n_samples, n_pixels, n_pixels) -> (n_pixels*n_pixels, n_samples)
    # y: (n_samples, 1) -> (1, n_samples)
    train_X_reshape = train_X.reshape(train_X.shape[0], train_X.shape[1] * train_X.shape[2]).T
    train_y_reshape = train_y.reshape(1, train_y.shape[0])
    test_X_reshape = test_X.reshape(test_X.shape[0], test_X.shape[1] * test_X.shape[2]).T
    test_y_reshape = test_y.reshape(1, test_y.shape[0])
    
    #%%
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
    # train model
    hidden_layers = np.array([32, 32, 32])
    my_layer = FC()
    my_layer.n_iter = 1000
    my_layer.stop_th = 30
    my_layer.verbose = 10
    parameters = my_layer.train_nn(train_X_reshape/255, y_train, 
                                   hidden_layers,
                                   test_X_reshape/255, y_test)
    # I reached 0.89965 at best