#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct  3 17:50:41 2020

@author: zhe
"""

# Machine Learning HW2 Ridge Regression

import matplotlib.pyplot as plt
import numpy as np

# Parse the file and return 2 numpy arrays
def load_data_set(filename):
    dataset = open(filename,'r').read().splitlines()
    x = [[]]
    count = []
    index = 0
    for line in dataset:
        temp = []
        count.append(0)
        for num in line.split():
            count[index] = count[index] + 1
            temp.append(float(num))
        x.append(temp)
        index = index + 1
    x = np.vstack(x[1:201])
    y = x[:,-1]
    x = x[:,0:-2]
    return x, y


# Split the data into train and test examples by the train_proportion
# i.e. if train_proportion = 0.8 then 80% of the examples are training and 20%
# are testing
def train_test_split(x, y, train_proportion):
    trainIndex = int(train_proportion * x.shape[0])
    x_train = x[0:trainIndex,:]
    y_train = y[0:trainIndex]
    x_test = x[trainIndex:-1,:]
    y_test = y[trainIndex:-1]
    return x_train, x_test, y_train, y_test

# Find theta using the modified normal equation, check our lecture slides
# Note: lambdaV is used instead of lambda because lambda is a reserved word in python
def normal_equation(x, y, lambdaV):
    return np.dot(np.linalg.inv(x.T.dot(x) + lambdaV*np.identity(x.shape[1])),x.T.dot(y))



# Given an array of y and y_predict return loss
def get_loss(y, y_predict):
    return sum((y_predict-y)**2)/y.shape[0]

# Given an array of x and theta predict y
def predict(x, theta):
    return np.dot(x,theta[:])
    

# Find the best lambda given x_train and y_train using 4 fold cv
def cross_validation(x_train, y_train, lambdas):
    folds = 4
    count = 0
    numVals = x_train.shape[0]
    valid_losses = np.zeros(len(lambdas))
    training_losses = np.zeros(len(lambdas))
    for lambdaV in lambdas:
        
        for i in range(folds):
            
            x_valid = x_train[int(i*numVals/4):int((i+1)*numVals/4),:]
            y_valid = y_train[int(i*numVals/4):int((i+1)*numVals/4)]
            x_fold = np.concatenate((x_train[0:int(i*numVals/4),:],x_train[int((i+1)*numVals/4):numVals,:]))
            y_fold = np.concatenate((y_train[0:int(i*numVals/4)],y_train[int((i+1)*numVals/4):numVals]))
            
            beta = normal_equation(x_fold,y_fold,lambdaV)
            training_losses[count] = training_losses[count] + get_loss(y_fold,predict(x_fold,beta))
            valid_losses[count] = valid_losses[count] + get_loss(y_valid,predict(x_valid,beta))
            
        training_losses[count] = training_losses[count]/folds
        valid_losses[count] = valid_losses[count]/folds
        count = count + 1

    return np.array(valid_losses), np.array(training_losses)


    
# Calcuate the l2 norm of a vector    
def l2norm(vec):
    return np.sqrt(sum(vec**2))

#  show the learnt values of Î² vector from the best Î»

def bar_plot(beta):
    plt.bar(range(1,len(beta)+1),beta)
    plt.show()
    
    
    
    

if __name__ == "__main__":

    # step 1
    # If we don't have enough data we will use cross validation to tune hyperparameter
    # instead of a training set and a validation set.
    x, y = load_data_set("dataRidge.txt") # load data
    x_train, x_test, y_train, y_test = train_test_split(x, y, 0.8)
    # Create a list of lambdas to try when hyperparameter tuning
    lambdas = [2**i for i in range(-3, 9)]
    lambdas.insert(0, 0)
    # Cross validate
    valid_losses, training_losses = cross_validation(x_train, y_train, lambdas)
    # Plot training vs validation loss
    plt.plot(lambdas[1:], training_losses[1:], label="training_loss") 
    # exclude the first point because it messes with the x scale
    plt.plot(lambdas[1:], valid_losses[1:], label="validation_loss")
    plt.legend(loc='best')
    plt.xscale("log")
    plt.yscale("log")
    plt.title("lambda vs training and validation loss")
    plt.show()

    best_lambda = lambdas[np.argmin(valid_losses)]


    # step 2: analysis 
    normal_beta = normal_equation(x_train, y_train, 0)
    best_beta = normal_equation(x_train, y_train, best_lambda)
    large_lambda_beta = normal_equation(x_train, y_train, 512)
    normal_beta_norm = l2norm(normal_beta)# your code get l2 norm of normal_beta
    best_beta_norm = l2norm(best_beta)# your code get l2 norm of best_beta
    large_lambda_norm = l2norm(large_lambda_beta)# your code get l2 norm of large_lambda_beta
    print(best_lambda)
    print("L2 norm of normal beta:  " + str(normal_beta_norm))
    print("L2 norm of best beta:  " + str(best_beta_norm))
    print("L2 norm of large lambda beta:  " + str(large_lambda_norm))
    print("Average testing loss for normal beta:  " + str(get_loss(y_test, predict(x_test, normal_beta))))
    print("Average testing loss for best beta:  " + str(get_loss(y_test, predict(x_test, best_beta))))
    print("Average testing loss for large lambda beta:  " + str(get_loss(y_test, predict(x_test, large_lambda_beta))))
    
    
    # step 3: visualization
    bar_plot(best_beta)


    
