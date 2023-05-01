# -*- coding: utf-8 -*-
"""
Created on Mon Apr 10 12:34:16 2023

@author: gbonnet
"""

import numpy as np
import matplotlib.pyplot as plt
import random

class my_metrics :
    
    def my_accuracy_score(self, y_true, y_pred):
        return np.sum(y_true == y_pred)/len(y_true)

    def my_log_loss(self, y_true, y_pred):
        return np.mean(-1 * (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))
    
class FC(my_metrics) :
    
    def __init__(self):
        self.learning_rate = 0.01
        self.n_epochs = 100 
        self.stop_th = 10 # early_stopping: training stops when loss does not improve for stop_th epochs in a row
        self.verbose = 1 # number of epoch you want to print the loss and accuracy
        self.batch_size = 32 # size of the minibatch
        
    def initialization(self, dimensions: np.ndarray) -> dict:
        '''
        Initialize the weight and biaise parameters W and b of the network.
        Parameters
        ----------
        dimensions: np.array, contains the the dimension of each layer,
                    dimensions[0] being the dimension of the input, 
                    e.g. the dimension of the first sample element
                    dimensions[1] being the number of unit of the first layer
                    dimensions[2] being the number of unit of the second layer, ...

        Returns
        -------
        parameters: dictionnary, key being the Wl and bl parameteres
                    and values being the values of these parameters.
                    Thus it contains the values of the weight and biaises parameters
        '''
        parameters = {}
        for l in range(1, self.N_layers+1) : # loop on the number of layers
            parameters['W' + str(l)] = np.random.randn(dimensions[l], dimensions[l - 1])
            parameters['b' + str(l)] = np.random.randn(dimensions[l], 1)
        return parameters
    
    def forward_propagation(self, X: np.ndarray, parameters: dict) -> dict:
        '''
        Propagates the weights and biaises on the input values X 
        and applies the activation functions for each layer.
        Parameters
        ----------
        X : np.ndarray, input data
        parameters : dict, weights and biaises parameters

        Returns
        -------
        activations: dict, activation function outputs for each layer
        '''
        activations = {'A0': X} # the 0th activation function output correponds to the input X
        for l in range(1, self.N_layers+1) : # loop on the layers
            # Z_l has a shape (n_unit_l, n_sample)
            # Z_l = W_l * A_l-1 + b_l
            Z = parameters['W' + str(l)].dot(activations['A' + str(l - 1)]) + parameters['b' + str(l)]
            # be carefull, this is activation function dependent
            # activation_l has a shape (n_unit_l, n_sample)
            # => activation_l-1 has a shape (n_unit_l-1, n_sample)
            # A_l = 1/(1+np.exp(-Z_l))
            activations['A'+str(l)] = 1 / (1 + np.exp(-Z))
        
        return activations
    
    def back_propagation(self, y: np.ndarray, activations: dict, parameters: dict) -> dict:
        '''
        does the back propagation i.e. computes the gradient 
        of the weights W and biaises b for each layer l.
        Parameters
        ----------
        y: np.ndarray, target
        activations: dict, activation functions
        parameters: dict, weights and biaises
        Returns
        -------
        gradients: dict, gradients of the weights W and biaises b
        '''
        m = y.shape[1] # Number of sample
        # dZ of layer l has a shape of (n_unit_l, n_sample)
        # activation of layer l-1 has a shape of (n_unit_l-1, n_sample)
        # its transpose has thus a shape of (n_sample, n_unit_l-1)
        # thus the matricial product of dZ_l times activations_l-1.transpose
        # has a shape of (n_unit_l, n_unit_l-1)
        # this is the gradient shape, equal to the shape of the weights W
        # initialization: dZ_l_final
        dZ = activations['A' + str(self.N_layers)] - y
        gradients = {}
        for l in reversed(range(1, self.N_layers + 1)): # loop on the layers from the last to the first
            gradients['dW' + str(l)] = (1 / m) * np.dot(dZ, activations['A' + str(l-1)].T)
            gradients['db' + str(l)] = (1 / m) * np.sum(dZ, axis=1, keepdims=True)
            if l > 1 : # dZ0 is meaningless
                # dZ_l-1 = dot(W_l.T , dZ_l) * A_l-1 * (1 - A_l-1)
                dZ = np.dot(parameters['W' + str(l)].T, dZ) * activations['A' + str(l - 1)] \
                    * (1 - activations['A' + str(l - 1)])
        
        return gradients
    
    def update(self, gradients: dict, parameters: dict) -> dict:
        '''
        updates the parameters w and b with the gradients
        Parameters
        ----------
        gradients : dict, gradients of w and b
        parameters : dict, weight w and biaises b
        Returns
        -------
        parameters: dict, weight w and biaises b updated
        '''
        # the parameters are updated after 
        # having forward and backward all the sample
        for l in range(1, self.N_layers + 1): # loop on the layers
            # w_new = w_old - learning_rate * d_w_old
            parameters['W' + str(l)] = parameters['W' + str(l)] - self.learning_rate * gradients['dW' + str(l)]
            parameters['b' + str(l)] = parameters['b' + str(l)] - self.learning_rate * gradients['db' + str(l)]
            
        return parameters
    
    def predict(self, X: np.ndarray, parameters: dict) -> np.ndarray:
        '''
        computes the outputs that you get for a certain input X 
        and parameters parameters W and b
        Parameters
        ----------
        X : np.ndarray, input data
        parameters : dict, weights and biaises
        Returns
        -------
        A_final: np.ndarray, predicted outputs of 0 and 1
        '''
        activations = self.forward_propagation(X, parameters)
        A_final = activations['A' + str(self.N_layers)]
        return A_final >= 0.5
    
    def early_stop_inf(self, f_true_test: list, stop: int) -> int:
        '''
        early stopping function for metrics that should increase when 
        the prediction improves, like e.g. accuracy.
        Parameters
        ----------
        f_true_test : list, contains the metrics previously obtained
        stop : int, number of times the prediction can not improve in a row
        Returns
        -------
        stop: int, update stop
        '''
        if f_true_test[-1] < f_true_test[-2-stop] :
            stop += 1
            print('new test:',f_true_test[-1],' < ',f_true_test[-2-stop],'old test')
        else :
            stop = 0
        return stop
    
    def early_stop_sup(self, f_true_test: list, stop: int) -> int:
        '''
        early stopping function for metrics that should decrease when 
        the prediction improves.
        Parameters
        ----------
        f_true_test : list, contains the metrics previously obtained
        stop : int, number of times the prediction can not improve in a row
        Returns
        -------
        stop: int, update stop
        '''
        if f_true_test[-1] > f_true_test[-2-stop] :
            stop += 1
            print('new test:',f_true_test[-1],' > ',f_true_test[-2-stop],'old test')
        else :
            stop = 0
        return stop
                
    def set_batch(self, N_samples: int, batch_size: int):
        '''
        this function builds the indices in order to generate the minibatches
        Parameters
        ----------
        N_samples : int, total number of samples in train
        batch_size : int, size of batch i.e. number of samples inside 1 batch
        Returns
        -------
        ind : indices of samples disordered
        N_batch : int, number of batches
        '''
        N_batch = N_samples // batch_size
        ind = random.sample(range(N_samples),N_samples)
        return ind, N_batch
    
    def train_nn(self, X: np.ndarray, y: np.ndarray, hidden_layers: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray):
        '''
        this function trains the fully connected neural network.
        Parameters
        ----------
        X : np.ndarray, input for train, (dimension, n_samples)
        y : np.ndarray, outputs (dimensions, n_samples)
        hidden_layers : np.ndarray, number of units inside each layer
        X_test : np.ndarray, input for test, (dimension, n_samples)
        y_test : np.ndarray, output for test (dimensions, n_samples
        Returns
        -------
        parameters: dict, weight and biaises parameters of the trained network
        measure: dict, loss and accuracy for the train and test set at all epochs
        '''        
        np.random.seed(0) # fix the randomness of the initialization
        # add the dimension of the problem as the first element of dimensions,
        # e.g. if each sample s in a 2 dimensional vector then X.shape[0] = 2
        # and then w_0 will be of shape (n_unit_1, 2)
        dimensions = np.insert(hidden_layers, 0, X.shape[0]) 
        dimensions = np.append(dimensions, y.shape[0])
        self.N_layers = len(dimensions) - 1 # total number of layers
        self.N_samples_train = y.shape[1] # total number of samples in train
        self.N_samples_test = y_test.shape[1] # total number of samples in test
        # initialization of the weights and biaises parameters
        parameters = self.initialization(dimensions)
        # computes some metrics: loss and accuracy
        train_loss, train_acc = [], []
        test_loss, test_acc = [], []
        stop = 0 # initialize early stopping
        for i in range(self.n_epochs): # loop on the number of epochs
            # reorder the data after each epoch
            ind, N_batch = self.set_batch(self.N_samples_train, self.batch_size)
            for b in range(N_batch): # loop on the batches
                # indices of the samples of the batch b
                ind_b = ind[b*self.batch_size:(b+1)*self.batch_size]
                activations = self.forward_propagation(X[:,ind_b], parameters) # forward propagates
                gradients = self.back_propagation(y[:,ind_b], activations, parameters) # computes the gradients
                parameters = self.update(gradients, parameters) # updates the weights and biaises
            # train save results
            # loss
            activations = self.forward_propagation(X, parameters) # forward propagates
            train_loss.append(self.my_log_loss(y.flatten(), activations['A' + str(self.N_layers)].flatten()))
            # accuracy
            y_pred = self.predict(X, parameters)
            train_acc.append(self.my_accuracy_score(y.flatten(), y_pred.flatten()))
            # test save results
            # loss
            activations_test = self.forward_propagation(X_test, parameters)
            test_loss.append(self.my_log_loss(y_test.flatten(), activations_test['A' + str(self.N_layers)].flatten()))
            # accuracy
            y_pred_test = self.predict(X_test, parameters)
            test_acc.append(self.my_accuracy_score(y_test.flatten(), y_pred_test.flatten()))
            # early stopping
            if i > 1 :
                #stop = self.early_stop_inf(test_acc, stop)
                stop = self.early_stop_sup(test_loss, stop)
                if stop > self.stop_th :
                    break
            # print loss, accuracy epochs every verbose epoch
            if i % self.verbose == 0:
                print('epoch = ', i,', ',
                      'test_loss = {:.5f}'.format(test_loss[i]),', ',
                      'test_acc = {:.5f}'.format(test_acc[i]))
        # visualisation: plot of loss and accuracy for train and test
        fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(3, 3))
        ax[0].plot(train_loss, c='b', label='loss train')
        ax[0].plot(test_loss, c='r', label='loss test')
        ax[0].set_xlabel('epoch')
        ax[0].set_ylabel('loss')
        ax[0].legend()
        ax[1].plot(train_acc, c='b', label='acc train')
        ax[1].plot(test_acc, c='r', label='acc test')
        ax[1].set_xlabel('epoch')
        ax[1].set_ylabel('accuracy')
        ax[1].legend()
        plt.show()
        # set metrics in a dictionnary
        measure = {'train_loss': train_loss, 'train_acc': train_acc,
                   'test_loss': test_loss, 'test_acc': test_acc}
        
        return parameters, measure
    

    
    
if __name__ == '__main__' :
    print('test')
    
    