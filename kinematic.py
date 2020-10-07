import numpy as np
import matplotlib.pyplot as plt
import os

from pandas.compat import StringIO
from Load import*
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import probplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.activations import elu
from keras.constraints import maxnorm
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from keras.utils.vis_utils import plot_model

np.random.seed(123)

"""
#######################################################function description##############################################################
scale: scales the data and returns it.
plot_prediction: Plots CL, CP and their corresponding prediction.
plot_stat: Plots qqplot and histogram of the error.
prepare_data: separate the data in order to compute a cross vallidation.
build_k_indices: Computes an array of indices for Cross validation, after randomizing the order of the data.
cross_validation_reg: Perfoms a k-fold cross validation and returns the train and test mse for ridge linear regression.
model_training: Trains the neural network model and plots the error.
cv_regularization_degree: Compute the mse with the differente folder.
cross_validation_rs:Performs a K-fold cross-validation with scaled input and returns the train and test mse.
prediction: make a prediction using a reidge linear regression.
build_averageC_to_alpha and build_averageC_to_alpha2 : model construction for neural networks
model_training2: train the neural network and compute a validation set in order to evaluate the model
cross_validation: compute a cross validation for neural network
######################################################################################################################################### """


def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = np.shape(y)[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def prepare_data(X,Y,k_indices ,k):
    num_observations = 4100
    index_test = k_indices[k]
    index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
    X_train = X[index_train,:]
    X_test = X[index_test,:]
    Y_train = Y[index_train,:]
    Y_test = Y[index_test,:]
    return X_train,X_test,Y_train,Y_test

def cross_validation_reg( X_train, X_test, Y_train, Y_test, alpha_ ,degree_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    loss_te = 0
    loss_tr =0
    
    for t in range(200):
        X_train_tmp = X_train
        X_test_tmp  = X_test
        Y_train_tmp = Y_train[:,t]
        Y_test_tmp  = Y_test[:,t]
        if degree_>=2 :
            for i in range(2,degree_+1):
                X_train_tmp =  np.c_[X_train_tmp,np.power( X_train,i)]
                X_test_tmp  =  np.c_[X_test_tmp,np.power(X_test,i)]
                
            
        regr = Ridge(alpha=alpha_)
        regr.fit(X_train_tmp, Y_train_tmp)
        y_pred = regr.predict(X_test_tmp)
        y_pred_tr = regr.predict(X_train_tmp)
        loss_te += mean_squared_error(y_pred , Y_test_tmp)
        loss_tr += mean_squared_error(y_pred_tr , Y_train_tmp)
            
    return  loss_te/(200) ,loss_tr/(200)

def cv_regularization_degree(X,Y,alphas,degree):
    seed = 123
    k_fold = 4
    num_observations = 4100
    # split data in k fold
    
    # define lists to store the loss of training data and test data
    rmse_tr_deg = []
    rmse_te_deg = []
    best_alpha =  []
    # cross validation
    k_indices = build_k_indices(Y, k_fold, seed) 
    
    for degree_ in degree:
        
        rmse_tr = []
        rmse_te = []
        for alpha in alphas:
            rmse_tr_tmp = 0
            rmse_te_tmp = 0

            for k in range(k_fold):
                X_train,X_test,Y_train,Y_test = prepare_data(X,Y,k_indices ,k)
                loss_test , loss_train = cross_validation_reg( X_train, X_test, Y_train, Y_test, alpha,degree_)
                rmse_te_tmp +=(loss_test)
                rmse_tr_tmp +=(loss_train)
                
            rmse_tr.append(rmse_tr_tmp/k_fold)
            rmse_te.append(rmse_te_tmp/k_fold)
        rmse_tr_array=np.asarray(rmse_tr)
        rmse_te_array=np.asarray(rmse_te)
        index_min = np.argmin(rmse_te_array)
        
        rmse_tr_deg.append(rmse_tr_array[index_min])
        rmse_te_deg.append(rmse_te_array[index_min])
        best_alpha.append(alphas[index_min])
        
    return rmse_te_deg ,rmse_tr_deg,best_alpha


def prediction( X_train, X_test, Y_train , alpha_ =0.48 ,degree_ = 2):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    loss_te = 0
    loss_tr =0
    y = X_test[:,0]
    for t in range(200):
        X_train_tmp = X_train
        X_test_tmp  = X_test
        Y_train_tmp = Y_train[:,t]
        if degree_>=2 :
            for i in range(2,degree_+1):
                X_train_tmp =  np.c_[X_train_tmp,np.power( X_train,i)]
                X_test_tmp  =  np.c_[X_test_tmp,np.power(X_test,i)]
                
        regr = Ridge(alpha=alpha_)
        regr.fit(X_train_tmp, Y_train_tmp)
        y_pred = regr.predict(X_test_tmp)
        y = np.c_[ y ,y_pred ]
        y_pred_tr = regr.predict(X_train_tmp)
        loss_tr += mean_squared_error(y_pred_tr , Y_train_tmp)
            
    return  loss_te/(200) ,loss_tr/(200),y[:,1:]


# Train the model passed in argument and evalute the mse on the train and test set
def model_training2(model,X_train,X_test,Y_train,Y_test):
    early_stop = EarlyStopping(monitor ="val_loss", patience = 50 , verbose = 1)
    history = model.fit(X_train, Y_train, validation_split=0.2, epochs=1000 ,callbacks=[early_stop] , verbose=0)
    train_mse = model.evaluate(X_train, Y_train, verbose=0)
    test_mse = model.evaluate(X_test, Y_test, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_mse, test_mse))
    return train_mse , test_mse

# Define the most efficient architecture of the neural network to predict new kinematics alphas.
def build_averageC_to_alpha(): 
    #architecture
    model = Sequential()
    model.add(Dense(200 ,input_shape=(2,) ,activation ='elu'))
    model.add(Dense(200,activation ='elu'))
    model.add(Dense(200 , activation ='linear'))
    print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def build_averageC_to_alpha2(): 
    #architecture
    model = Sequential()
    model.add(Dense(32 ,input_shape=(2,) ,activation ='sigmoid'))
    model.add(Dense(64,activation ='sigmoid'))
    model.add(Dense(128,activation ='sigmoid'))
    model.add(Dense(200))
    print(model.summary())
    
    #optimiser 
    model.compile(loss='mean_squared_error',optimizer='adam')
    return model

def scale( X,min_ , max_ ):
    return (X - min_)/(max_-min_) 

def cross_validation(build_model,X,Y,K_fold ):
    seed = 123 ;
    num_observations = np.shape(X)[0]
    k_indices = build_k_indices(Y,K_fold,seed)
    train_mse = 0
    test_mse = 0
    for k in range(K_fold):
        index_test = k_indices[k]
        index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
        X_train = X[index_train,:]
        X_test = X[index_test,:]
        Y_train = Y[index_train,:]
        Y_test = Y[index_test,:]
        
        model = build_model()
        train_tmp ,test_tmp = model_training2(model,X_train,X_test,Y_train,Y_test)
        train_mse += train_tmp
        test_mse += test_tmp     
            
    train_mse /= K_fold 
    test_mse /= K_fold 
    return train_mse,test_mse