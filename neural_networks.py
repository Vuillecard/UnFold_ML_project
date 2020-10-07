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

import matplotlib.gridspec as gridspec

"""
#######################################################function description##############################################################
scale: scales the data and returns it.
plot_prediction: Plots CL, CP and their corresponding prediction.
plot_stat:Plots qqplot and histogram of the error.
build_k_indices: Computes an array of indices for Cross validation, after randomizing the order of the data.
cross_validation: Perfoms a k-fold cross validation and returns the train and test mse.
model_training: Trains the neural network model and plots the error.
cross_validation_rs:Performs a K-fold cross-validation with scaled input and returns the train and test mse
######################################################################################################################################### """
def scale( X,min_ , max_ ):
    return (X - min_)/(max_-min_) 


def plot_prediction(model , X_test , Y_test):
    y_pred = model.predict(X_test, verbose=0)
    N = 20
    fig = plt.figure(figsize = (20,5*(N//5))) 
    gs = gridspec.GridSpec((N+5)//5 , 5)
    h = 0
    for i in range(N):
        fig_ax = fig.add_subplot(gs[h//5, h%5])
        h += 1
        fig_ax.plot(y_pred[i,:].T,label ='NN')
        fig_ax.plot(Y_test[i,:],label='true')
        fig_ax.set_title(f"Prediction {i}")
        
        
        
        
def plot_stat(model , X_test , Y_test):
    n_row = np.shape(X_test)[0]
    n_col = np.shape(X_test)[1]
    
    y_pred = model.predict(X_test, verbose=0)
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    probplot((y_pred-Y_test).reshape((n_row*n_col,)),plot=ax1)
    ax2 = fig.add_subplot(122)
    ax2.hist((y_pred-Y_test).reshape((n_row*n_col,)), density = True, bins = 100)
    
    
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
    train_mse = 0
    test_mse = 0
    for k in range(K_fold):
        index_test = k_indices[k]
        index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
        X_train = data[index_train,0,:]
        X_test = data[index_test,0,:]
        Y_train = data[index_train,1,:]
        Y_test = data[index_test,1,:]
        
        model = build_model(summary = False)
        train_tmp ,test_tmp = model_training(model,X_train,X_test,Y_train,Y_test, plot = False)
        train_mse += train_tmp
        test_mse += test_tmp     
            
    train_mse /= K_fold 
    test_mse /= K_fold 
    return train_mse,test_mse


def model_training(model,X_train,X_test,Y_train,Y_test, patience_ = 50 , plot=True ):
    """Train the different models and return the MSE values"""
    early_stop = EarlyStopping(monitor ="val_loss", patience = patience_ , verbose = 1)
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=2000 ,callbacks=[early_stop] , verbose=0)
    train_mse = model.evaluate(X_train, Y_train, verbose=0)
    test_mse = model.evaluate(X_test, Y_test, verbose=0)
    print('MSE Train: %.3f | MSE Test: %.3f' % (train_mse, test_mse))
    if plot :
        # plot loss during training
        plt.title('Loss / Mean Squared Error')
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.legend()
        plt.show()
    return train_mse,test_mse

def cross_validation_rs(build_model, data,K_fold ,patience_=30):
    seed = 123 ;
    num_observations = np.shape(data)[0]
    k_indices = build_k_indices(data,K_fold,seed)
    train_mse = 0
    test_mse = 0
    for k in range(K_fold):
        index_test = k_indices[k]
        index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
        X_train = scale(data[index_train,0,:],95,-95)
        X_test = scale(data[index_test,0,:],95,-95)
        Y_train = data[index_train,1,:]
        Y_test = data[index_test,1,:]
        
        model = build_model(summary = False)
        train_tmp ,test_tmp = model_training(model,X_train,X_test,Y_train,Y_test,patience_)
        train_mse += train_tmp
        test_mse += test_tmp     
            
    train_mse /= K_fold 
    test_mse /= K_fold 
    return train_mse,test_mse
   