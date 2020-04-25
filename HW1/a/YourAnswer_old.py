import numpy as np
import random

def euclideanDistance(targetX, dataSet):
    """
    * 정답 설명
    targetX = (1,n)
    dataSet = (m,n)
    sqDiffMat = (m,n)
    distances = (m,1)
    """
    distances = 0
    
    ########################################################################################
    # TODO : Complete the code to obtain euclideanDistance between targetX and dataSet.
    #---------------------------------------WRITE YOUR CODE--------------------------------------------# 
    distances = np.linalg.norm(targetX - dataSet, axis=1)
    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ########################################################################################
    return distances

def getKNN(targetX, dataSet, labels, k):
    """
    * 정답 설명
    targetX = (1,n)
    dataSet = (m,n)
    labels = (m,1)
    k = (1)
    """
    ## targetX : target data
    ## dataset : other data
    ## labels
    ## k : number of neighbors
    
    # compute euclidean distance
    distances = euclideanDistance(targetX,dataSet)
    closest_data = 0
    
    ########################################################################################
    # TODO : Use the result of finding the distance between TargetX and other data,
    #              select the most out of k data closest to target data
    #---------------------------------------WRITE YOUR CODE--------------------------------------------#
    ids = np.argpartition(distances, k)
    closest_data = np.argmax(np.bincount(labels[ids[:k]]))
    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ########################################################################################
    
    return closest_data

def predictKNN(targetX, dataSet, labels, k):
    
    ## targetX : target data
    ## dataset : other data
    ## labels
    ## k : number of neighbors
    n = targetX.shape[0]
    
    ## predicted_array : array of the predicted labels for each target data using KNN, 
    ##                   having the same length as data.
    
    predicted_array = np.zeros((n,))
    
    for i in range(n):
        ########################################################################################
        # TODO : Using the result of closest data from getKNN,
        #              put the predicted label in the predicted array.
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#
        predicted_array[i] = getKNN(targetX[i], dataSet, labels, k)
        #--------------------------------------END OF YOUR CODE--------------------------------------------#
        ########################################################################################

    return predicted_array

class Softmax(object):
    def __init__(self):
        #self.Weights = None
        return
        
    def train(self, X_tr_data, Y_tr_data, X_val_data, Y_val_data, lr=1e-3, reg=1e-5, iterations=100, bs=128, verbose=False, weight=0):
        """
        Train this Softmax classifier using stochastic gradient descent.
        
        Inputs have D dimensions, and we operate on N examples.
        
        Inputs :
            - X_data : A numpy array of shape (N,D) containing training data.
            - Y_data : A numpy array of shape (N,) containing training labels;
                  Y[i]=c means that X[i] has label 0<=c<C for C classes.
            - lr : (float) Learning rate for optimization.
            - reg : (float) Regularization strength. 
            - iterations : (integer) Number of steps to take when optimizing. 
            - bs : (integer) Number of training examples to use at each step.
            - verbose : (boolean) If true, print progress during optimization.
        
        Returns :
            - A list containing the value of the loss function at each training iteration.
        """
        
        num_train, dim = X_tr_data.shape
        num_classes = np.max(Y_tr_data)+1
        self.Weights = 0.001*np.random.randn(dim, num_classes)
        
        if np.shape(weight)!=np.shape(0):
            self.Weights = weight
            
        for it in range(iterations):
            #X_batch = None
            #Y_batch = None
            
            #######################################################################################
            # TODO : Sample batch_size elements from the training data and their corresponding labels
            #              to use in this round of gradient descent.
            #              Store the data in X_batch and their corresponding labels in Y_batch; After sampling
            #              X_batch should have shape (dim, batch_size) and Y_batch should have shape (batch_size,)
            #
            #   Hint : Use np.random.choice to generate indicies.
            #             Sampling with replacement is faster than sampling without replacement.
            #---------------------------------------WRITE YOUR CODE--------------------------------------------#
            ids = np.arange(num_train)
            sample_ids = np.random.choice(ids, bs, replace=True)
            X_batch = X_tr_data[sample_ids[:]]
            Y_batch = Y_tr_data[sample_ids[:]]
            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ########################################################################################

            # Evaluate loss and gradient
            tr_loss, tr_grad = self.loss(X_batch, Y_batch, reg)

            
            
            # Perform parameter update
            ########################################################################################
            # TODO : Update the weights using the gradient and the learning rate
            #---------------------------------------WRITE YOUR CODE--------------------------------------------#
            self.Weights = self.Weights - lr * tr_grad
            #--------------------------------------END OF YOUR CODE--------------------------------------------#
            ########################################################################################

            if verbose and it % num_iters == 0:
                print ('Ieration %d / %d : loss %f ' % (it, num_iters, loss))
            
        
    
    def predict(self, X_data):
        """
        Use the trained weights of this softmax classifier to predict labels for data points.
        
        Inputs :
            - X : A numpy array of shape (N,D) containing training data.
            
        Returns :
             - Y_pred : Predicted labels for the data in X. Y_pred is a 1-dimensional array of length N, 
                        and each element is an integer giving the predicted class.
        """
        Y_pred = np.zeros(X_data.shape[0])
        
        ########################################################################################
        # TODO : Implement this method. Store the predicted labels in Y_pred
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#
        Y_pred = np.argmax(X_data.dot(self.Weights), axis=1)
        #--------------------------------------END OF YOUR CODE--------------------------------------------#
        ########################################################################################
        return Y_pred
    
    def get_accuracy(self, X_data, Y_data):
        """
        Use X_data and Y_data to get an accuracy of the model.
        
        Inputs :
            - X_data : A numpy array of shape (N,D) containing input data.
            - Y_data : A numpy array of shape (N,) containing a true label.
            
        Returns :
             - Accuracy : Accuracy of input data pair [X_data, Y_data].
        """
        accuracy = 0 
        
        ########################################################################################
        # TODO : Implement this method. Calculate an accuracy of X_data using Y_data and predict Func
        #---------------------------------------WRITE YOUR CODE--------------------------------------------#
        accuracy = np.sum(self.predict(X_data) == Y_data) / Y_data.shape[0]
        #--------------------------------------END OF YOUR CODE--------------------------------------------#
        #########################################################################################
        
        return accuracy
    
    def loss(self, X_batch, Y_batch, reg):
        return vectorized_softmax_loss(self.Weights, X_batch, Y_batch, reg)
    
def naive_softmax_loss(Weights,X_data,Y_data,reg):
    """
     Inputs have D dimension, there are C classes, and we operate on minibatches of N examples.
    
     Inputs :
         - Weights : A numpy array of shape (D,C) containing weights.
         - X_data : A numpy array of shape (N,D) contatining a minibatch of data.
         - Y_data : A numpy array of shape (N,) containing training labels; 
               Y[i]=c means that X[i] has label c, where 0<=c<C.
         - reg : Regularization strength. (float)
         
     Returns :
         - loss as single float
         - gradient with respect to Weights; an array of sample shape as Weights
     """
    
    # Initialize the loss and gradient to zero
    softmax_loss = 0.0
    dWeights = np.zeros_like(Weights)
    
    #########################################################################################
    # TODO : Compute the softmax loss and its gradient using explicit loops.
    #        Store the loss in loss and the gradient in dW.
    #        If you are not careful here, it is easy to run into numeric instability.
    #        Don't forget the regularization.
    #---------------------------------------WRITE YOUR CODE---------------------------------------------#
    output = X_data.dot(Weights)
    dim = Y_data.shape[0]
    class_num = output.shape[1]

    #Calculating the Li losses and adding to the softmax loss
    for i in range(dim):
        sum_prob = 0.0
        for j in range(class_num):
            sum_prob += np.exp(output[i, j])
        Pj = np.exp(output[i]) / sum_prob
        softmax_loss -= np.log(Pj[Y_data[i]])

        #Calculating the Gradient
        for j in range(class_num):
            if Y_data[i] == j:
                dWeights[:,j] += (Pj[j] - 1.0) * X_data[i].T
            else:
                dWeights[:,j] += Pj[j] * X_data[i].T
    softmax_loss /= dim
    dWeights /= dim

    #Calculating the L2 regularization
    reg_sum = 0.0
    for i in Weights:
        for j in i:
            reg_sum += j**2

    #We multiply 0.5 just so the derivative cancels it when calculating the gradient
    softmax_loss += 0.5 * reg * reg_sum
    dWeights += reg * Weights
    
    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    ########################################################################################
    
    return softmax_loss, dWeights

def vectorized_softmax_loss(Weights, X_data, Y_data, reg):
    softmax_loss = 0.0
    dWeights = np.zeros_like(Weights)

    ########################################################################################
    # TODO : Compute the softmax loss and its gradient using no explicit loops.
    #        Store the loss in loss and the gradient in dW.
    #        If you are not careful here, it is easy to run into numeric instability.
    #        Don't forget the regularization.
    #---------------------------------------WRITE YOUR CODE--------------------------------------------#
    output = X_data.dot(Weights)
    softmax_matrix = np.exp(output) / np.sum(np.exp(output), axis=1, keepdims=True)

    #Filter the matrix to use only the values where Syi was in the numerator
    softmax_matrix_filtered = np.choose(Y_data, softmax_matrix.T)
    softmax_loss = np.mean(-np.log(softmax_matrix_filtered)) + 0.5 * reg * np.sum(np.square(Weights))
    
    dim = Y_data.shape[0]
    #Subtract 1 for the case j = Yi
    softmax_matrix[range(dim), Y_data] -= 1
    dWeights = (X_data.T.dot(softmax_matrix)) / dim
    dWeights += reg * Weights

    #--------------------------------------END OF YOUR CODE--------------------------------------------#
    #########################################################################################
    
    return softmax_loss, dWeights





