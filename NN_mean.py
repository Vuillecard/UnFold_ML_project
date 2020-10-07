import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os

import keras
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.activations import elu
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

from sklearn.preprocessing import MinMaxScaler

from scipy.stats import probplot

from neural_networks import *
from helpers import *

np.random.seed(123)

"""
#######################################################function description##############################################################
plot_stat: Plots qqplot and histogram of the error.
build_k_indices: Computes an array of indices for Cross validation, after randomizing the order of the data for the mean output.
cross_validation: Perfoms a k-fold cross validation and returns the train and test mse for the mean output.
cross_validation_sc: Performs a K-fold cross-validation with scaled input and returns the train and test mse for the mean output.
######################################################################################################################################### """
def plot_stat(model , X_test , Y_test):
    n_row = np.shape(X_test)[0]
    n_col = np.shape(X_test)[1]
    y_pred = model.predict(X_test, verbose=0)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    probplot((y_pred-Y_test).reshape((n_row,)),plot=ax1)
    ax2 = fig.add_subplot(122)
    ax2.hist((y_pred-Y_test).reshape((n_row,)), density = True, bins = 20)
    
    
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)




def cross_validation(build_model, data,K_fold ):
    seed = 123 ;
    num_observations = np.shape(data)[0]
    k_indices = build_k_indices(data,K_fold,seed)
    Y= np.mean(data[:,1,:] , axis =1).reshape((num_observations,1))
    train_mse = 0
    test_mse = 0
    for k in range(K_fold):
        index_test = k_indices[k]
        index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
        X_train = data[index_train,0,:]
        X_test = data[index_test,0,:]
        Y_train = Y[index_train]
        Y_test = Y[index_test]
        
        model = build_model(summary = False)
        train_tmp ,test_tmp = model_training(model,X_train,X_test,Y_train,Y_test, plot = False)
        train_mse += train_tmp
        test_mse += test_tmp     
            
    train_mse /= K_fold 
    test_mse /= K_fold 
    return train_mse,test_mse



def cross_validation_sc(build_model, data,K_fold ):
    seed = 123 ;
    num_observations = np.shape(data)[0]
    k_indices = build_k_indices(data,K_fold,seed)
    Y= np.mean(data[:,1,:] , axis =1).reshape((num_observations,1))
    train_mse = 0
    test_mse = 0
    for k in range(K_fold):
        index_test = k_indices[k]
        index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
        X_train = scale(data[index_train,0,:],-95,95)
        X_test = scale(data[index_test,0,:],-95,95)
        Y_train = Y[index_train]
        Y_test = Y[index_test]
        
        model = build_model()
        train_tmp ,test_tmp = model_training(model,X_train,X_test,Y_train,Y_test, plot = False)
        train_mse += train_tmp
        test_mse += test_tmp     
            
    train_mse /= K_fold 
    test_mse /= K_fold 
    return train_mse,test_mse