#-*- coding: utf-8 -*-
import numpy as np
import random
import os
import matplotlib.pyplot as plt
from collections import OrderedDict
from utils import numerical_gradient

def softmax(x):
    
    """
    Make Softmax function.
    Be careful not to make NaN.(preventing overflow)
    Implement considering Batch size(N).
    
    Inputs : 
        - x : vector with dimension (N,D)
    
    Output : 
        - softmax_output : Softmax result with dimension (N,D)
    """
    
    softmax_output = None
    
    #########################################################################################################
    #------------------------------------------WRITE YOUR CODE----------------------------------------------#
    x = x - np.max(x, axis=1, keepdims=True)
    tmp = np.exp(x)
    softmax_output = tmp / np.sum(tmp, axis=1, keepdims=True)
    #-----------------------------------------END OF YOUR CODE----------------------------------------------#
    #########################################################################################################
    
    return softmax_output




def cross_entropy_loss(score, target, weights, regularization):
    
    """    
    Make cross_entropy_loss function.
    Let 0 not to be an input to log using delta.
    Implement considering Batch size(N).
    Implement considering L2 Regularization using Weights.
    Calculate L2 Regularization with multiplying 0.5. (advantage when differentiating)
    i.e regularization strength * 0.5 * L2 Regularization
    
    Inputs : 
        - score : vector with dimension (N, D)
        - target : vector with dimension (N, D) (One-hot encoding)
        - weights : set of weight matrices (dictionary)
        - regularization : a number between 0 to 1, which sets regularization
    
    Output : 
        - loss : loss value which is a scalar    
    """
    
    delta = 1e-9
    batch_size = target.shape[0]
    data_loss = 0
    reg_loss = 0
    loss = None
    
    #########################################################################################################
    #------------------------------------------WRITE YOUR CODE----------------------------------------------#
    data_loss = np.mean(-np.log(score[range(batch_size), np.argmax(target, axis=1)] + delta))
    reg_loss = sum([0.5 * regularization * np.sum(np.square(weights[w])) for w in weights])
    #-----------------------------------------END OF YOUR CODE----------------------------------------------#
    #########################################################################################################
    
    loss = data_loss + reg_loss
    
    return loss
    

    

class OutputLayer:
    
    """
    Make Ouput Layer class which calculates Cross-entropy loss, using Softmax.
    Use Softmax() and cross_entropy_loss, which are previously made.
    Consider the calculating process of forward, backward.
    Be careful that output of softmax and target label are used in backpropagation.
    
    forward() : 
        - x : vector with dimension (N,D)
        - y : vector with dimension (N, # of Label) 
        - return : softmax loss
    
    backward() : 
        - dout : delta from backpropagation, delta = 1 as it is output layer
        - return : dx    
    """
    
    def __init__(self, weights, regularization):
        self.loss = None           # loss value
        self.output_softmax = None # Output of softmax
        self.target_label = None   # Target label (one-hot vector)
        self.weights = weights
        self.regularization = regularization
        
    def forward(self, x, y):
    
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.target_label = y
        self.output_softmax = softmax(x)
        self.loss = cross_entropy_loss(self.output_softmax, self.target_label, self.weights, self.regularization)
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
    
        return self.loss
    
    def backward(self, dout=1):
        
        bt_size = self.target_label.shape[0]
        dx = None
        
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dx = self.output_softmax
        dx[range(bt_size), np.argmax(self.target_label, axis = 1)] -= 1
        dx *= dout / bt_size
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        
        return dx
    
    
class ReLU:
    
    """
    Implement RELU.
    Consider the calculating process of forward, backward.
    Be careful that mask information used in forward process is utilized in backpropagation.
    
    forward() : 
        - x : vector with dimension (N,D)
        - return : ReLU output
    
    backward() : 
        - dout : delta from backpropagation
        - return : dx    
    """
    
    def __init__(self):
        
        self.mask = None
        
    def forward(self, x):
        
        out = None
    
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.mask = (x > 0) * 1
        out = np.abs(x) * self.mask
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
    
        return out
    
    def backward(self, dout):
    
        dx = None
        
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dx = self.mask*dout
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        
        return dx
    
class Sigmoid:
    
    """
    Implement Sigmoid.
    Consider the calculating process of forward, backward.
    Be careful that the results from forward process is utilized in backpropagation.
    
    forward() : 
        - x : vector with dimension (N,D)
        - return : sigmoid output
    
    backward() : 
        - dout : delta from backpropagation
        - return : dx    

    """
    
    def __init__(self):
        self.out = None
        
    def forward(self, x):
    
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.out = 1 / (1 + np.exp(-x))
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
    
        return self.out
    
    def backward(self, dout):
        
        dx = None
        
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        dx = self.out * (1 - self.out) * dout
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        
        return dx
    
    
    
class Affine:
    
    """
    Implement Affine layer.
    Affine layer is calculating one neuron's weighted sum like Y = np.dot(X, W) + B (X: input, W: weight, B: bias)
    Consider the the calculating process of forward, backward.
    Consider the Batch size(N).
    Be careful that W,b, and x are used in backward.
    You have to calculate self.dW & self.db in backward function.
    self.dW & self.db will be used in gradient function of TwolayerNet
    
    forward() : 
        - x : vector with dimension (N,D)
        - return : Affine output
    
    backward() : 
        - dout : delta from backpropagation
        - return : dx    
    
    """
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
        
    def forward(self, x):
        
        out = None
    
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.x = x
        out = self.x.dot(self.W) + self.b
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
    
        return out
    
    def backward(self, dout):
        
        dx = None
        
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.dW = self.x.T.dot(dout)
        self.db = np.sum(dout, axis=0)
        dx = dout.dot(self.W.T)
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        
        return dx
    
    
    
    
from collections import OrderedDict
class TwoLayerNet:
    
    """
    __init__() : 
        - A function initializing weight and bias
        - A function making layers with initialized weight and bias
        
    predict() : 
        - A function performing forward propagation in neural network with input data(x)

    loss() : 
        - A function calculating loss using the results from forward propagation of neural network with input data(x)
        
    accuracy() :
        - A function obtaining accuracy using the results from input data(x) and true label(y)
        
    numerical_gradient() :
        - A function obtaining numerical gradient using input data(x) and true label(y)
        - Utilized to compare with gradient using backpropagation
        
    gradient():
        - A function performing backpropagation using input data(x) and true label(y)
    
    """

    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01, regularization = 0.0):

        # Weight initialization
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size) 
        self.params['b2'] = np.zeros(output_size)

        self.weights = {}
        self.weights['W1'] = self.params['W1']
        self.weights['W2'] = self.params['W2']
        
        self.reg = regularization

        # Layer generation
        self.layers = OrderedDict() # information about OrderedDict (https://pymotw.com/2/collections/ordereddict.html)
        
        #########################################################################################################
        # Implement TwoLayerNet.
        # Implement Neural Network structure as follow
        # [ Input => Fully Connected => ReLU => Fully Connected ] => OutputLayer  
        # Implement using previously made Class(Affine, ReLU) and weights from Weight initialization
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        self.lastLayer = OutputLayer(self.weights, self.reg)

    def predict(self, x):

        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, y):
        score = self.predict(x)
        return self.lastLayer.forward(score, y)

    

    def accuracy(self, x, y):

        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, y):

        loss_W = lambda W: self.loss(x, y)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        
        if self.reg != 0.0:
                       
            #########################################################################################################
            # Implement Regularization part when obtaining Gradient.
            # We already implemnented the computing process of numerical gradient in utils.py, 
            # so you only need to consider gradient of regularization term.
            #########################################################################################################
            #------------------------------------------WRITE YOUR CODE----------------------------------------------#
            grads['W1'] += self.reg * self.params['W1']
            grads['b1'] += self.reg * self.params['b1']
            grads['W2'] += self.reg * self.params['W2']
            grads['b2'] += self.reg * self.params['b2']
            #-----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################
            
        return grads

        

    def gradient(self, x, y):

        # forward
        self.loss(x, y)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        
        if self.reg != 0.0:
            
            #########################################################################################################
            # Implement the effect of regularization when obtaining each gradient of weight and bias.
            #########################################################################################################
            #------------------------------------------WRITE YOUR CODE----------------------------------------------#
            grads['W1'] += self.reg * self.params['W1']
            grads['b1'] += self.reg * self.params['b1']
            grads['W2'] += self.reg * self.params['W2']
            grads['b2'] += self.reg * self.params['b2']
            #-----------------------------------------END OF YOUR CODE----------------------------------------------#
            #########################################################################################################

        return grads
    
    
    
    
    
class ThreeLayerNet:
    
    """
    
    __init__() : 
        - A function initializing weight and bias
        - A function making layers using initialized weight and bias
    
    predict() : 
        - A function performing forward propagation of neural network with input data(x)
        
    loss() : 
        - A function calculating loss using the results from forward propagation of neural network with input data(x)
        
    accuracy() :
        - A function obtaining accuracy using the results from input data(x) and true label(y)
        
    numerical_gradient() :
        - A function obtaining numerical gradient using input data(x) and true label(y)
        - Utilized to compare with gradient using backpropagation
        
    gradient():
        - A function performing backpropagation using input data(x) and true label(y)    
    
    """

    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, weight_init_std = 0.01, regularization = 0.0):

        
        #########################################################################################################
        # Implement ThreeLayerNet applying TwoLayerNet
        # Implement Neural Network structure as follow:
        #[ Input => Fully Connected => ReLU => Fully Connected => ReLU => Fully Connected ] => OutputLayer
        # Implement using the Class previously made
        # * Consider the elements to be changed as Hidden Layer increase (e.g. addition of Weight and bias, number of weight in Hidden Layers, 
        #  Weight update) *
        # Use hidden_size1 and hidden_size2 as variables of Hidden Layer
        #########################################################################################################
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
         # Weight initialization
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size1)
        self.params['b1'] = np.zeros(hidden_size1)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size1, hidden_size2)
        self.params['b2'] = np.zeros(hidden_size2)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size2, output_size)
        self.params['b3'] = np.zeros(output_size)

        self.weights = {}
        self.weights['W1'] = self.params['W1']
        self.weights['W2'] = self.params['W2']
        self.weights['W3'] = self.params['W3']
        
        self.reg = regularization

        # Layer generation
        self.layers = OrderedDict()

        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.layers['Relu2'] = ReLU()
        self.layers['Affine3'] = Affine(self.params['W3'], self.params['b3'])
        
        self.lastLayer = OutputLayer(self.weights, self.reg)
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################
        return

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
            
        return x

    def loss(self, x, y):        
        score = self.predict(x)
        
        return self.lastLayer.forward(score, y)

    

    def accuracy(self, x, y):
        score = self.predict(x)
        score = np.argmax(score, axis=1)
        if y.ndim != 1 : y = np.argmax(y, axis=1)
        accuracy = np.sum(score == y) / float(x.shape[0])

        return accuracy

        

    def gradient(self, x, y):
        # forward
        self.loss(x, y)
        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()

        for layer in layers:
            dout = layer.backward(dout)

        grads = {}
        #########################################################################################################
        # Implement the part of obtaining gradient of each weight and bias
        #########################################################################################################
        #------------------------------------------WRITE YOUR CODE----------------------------------------------#
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        grads['W3'], grads['b3'] = self.layers['Affine3'].dW, self.layers['Affine3'].db
        
        if self.reg != 0.0:
            grads['W1'] += self.reg * self.params['W1']
            grads['b1'] += self.reg * self.params['b1']
            grads['W2'] += self.reg * self.params['W2']
            grads['b2'] += self.reg * self.params['b2']
            grads['W3'] += self.reg * self.params['W3']
            grads['b3'] += self.reg * self.params['b3']
        #-----------------------------------------END OF YOUR CODE----------------------------------------------#
        #########################################################################################################

        return grads
    
    
    
    
