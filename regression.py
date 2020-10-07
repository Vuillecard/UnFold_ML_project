import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import probplot
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_validate

"""
#######################################################function description##############################################################
generate_prediction_n: Trains the linear regression with a window of size n, a degree_ of augmented features, and an alpha_ corresponding
                       to the weight of the regularizing term, it returns the prediction.
K_fold validation for a window size of n:
            cv_n:Computes K-fold cross-validation, for different widows of size n
            cross_validation_n: Computes MSE for a given windiw size n
K_fold validation for Ridge regression:
            cross_validation_reg:Computes MSE for a given value alpha of the regularizing term
            cv_regularization: Computes k-fold cross-validation, for different values of alpha
K_fold validation for Augmented inputs:
            cv_regularization_degree: Computes K-fold cross-validation, for different degree of the augmented input matrix
AVERAGE CP AND AVERAGE CL:
            prepare_data_average:Creates the different set of inputs and outputs and prepare them as the outputs are the mean CL and CP
            average_cv_reg: Computes the MSE for a given alpha
            average_cv_regularization: Computes k-fold cross-validation, for different values of alpha, the weight of the regularizing term
            simple_cv:K-fold cross validation for the averaged CL and CP
######################################################################################################################################### """
def generate_prediction_n(  X_train , X_test , Y_train , Y_test, alpha_ ,n,degree_ ):
    # get k'th subgroup in test, others in train
    m = int(n/2)
    y = Y_test[:,1]
    for t in range(m,np.shape(X_train)[1]-m):
        X_train_tmp = X_train[:,t-m:t+m+1]
        X_test_tmp  = X_test[:,t-m:t+m+1]
        Y_train_tmp = Y_train[:,t]
        Y_test_tmp  = Y_test[:,t]
        if degree_>=2 :
            for i in range(2,degree_+1):
                X_train_tmp =  np.c_[X_train_tmp,np.power( X_train[:,t-m:t+m+1],i)]
                X_test_tmp  =  np.c_[X_test_tmp,np.power(X_test[:,t-m:t+m+1],i)]
                
            
        regr = Ridge(alpha=alpha_)
        regr.fit(X_train_tmp, Y_train_tmp)
        y_pred = regr.predict(X_test_tmp)
        y = np.c_[y, y_pred]
        y_pred_tr = regr.predict(X_train_tmp)    
            
    return  y[:,1:]



def build_k_indices_n(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = np.shape(y)[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval] for k in range(k_fold)]
    return np.array(k_indices)

def prepare_data_xyw1(data,k_indices ,k):
    num_observations = 4100
    index_test = k_indices[k]
    index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
    X_train = data[index_train,0,:]
    X_test = data[index_test,0,:]
    Y_train = data[index_train,1,:]
    Y_test = data[index_test,1,:]
    return X_train,X_test,Y_train,Y_test
######################################### K_fold validation for a window size of n ###########################################

def cross_validation_n( X_train, X_test, Y_train, Y_test, n):
    # get k'th subgroup in test, others in train
    loss_te = 0
    loss_tr =0
    m = int(n/2)
    for t in range(m,np.shape(X_train)[1]-m):
        regr = linear_model.LinearRegression()
        regr.fit(X_train[:,t-m:t+m+1], Y_train[:,t])
        y_pred = regr.predict(X_test[:,t-m:t+m+1])
        y_pred_tr = regr.predict(X_train[:,t-m:t+m+1])
        loss_te += mean_squared_error(y_pred , Y_test[:,t])
        loss_tr += mean_squared_error(y_pred_tr , Y_train[:,t])
    return  loss_te/(200-(2*m)) ,loss_tr/(200-(2*m))
def cv_n(data,window):
    seed = 12
    degree = 7
    k_fold = 4
    lambda_ = 0
    num_observations = 4100
    # split data in k fold
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation

    
    k_indices = build_k_indices_n(data, k_fold, seed) 
    for n in window:
        rmse_tr_tmp = 0
        rmse_te_tmp = 0
        
        for k in range(k_fold):
            X_train,X_test,Y_train,Y_test = prepare_data_xyw1(data,k_indices ,k)
            loss_test , loss_train = cross_validation_n( X_train, X_test, Y_train, Y_test, n)
            rmse_te_tmp +=(loss_test)
            rmse_tr_tmp +=(loss_train)
        rmse_tr.append(rmse_tr_tmp/k_fold)
        rmse_te.append(rmse_te_tmp/k_fold) 
        
    return rmse_te ,rmse_tr
######################################### K_fold validation for Ridge regression###########################################
def cross_validation_reg( X_train, X_test, Y_train, Y_test, alpha_ ,n,degree_):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    loss_te = 0
    loss_tr =0
    m = int(n/2)
    for t in range(m,np.shape(X_train)[1]-m):
        X_train_tmp = X_train[:,t-m:t+m+1]
        X_test_tmp  = X_test[:,t-m:t+m+1]
        Y_train_tmp = Y_train[:,t]
        Y_test_tmp  = Y_test[:,t]
        if degree_>=2 :
            for i in range(2,degree_+1):
                X_train_tmp =  np.c_[X_train_tmp,np.power( X_train[:,t-m:t+m+1],i)]
                X_test_tmp  =  np.c_[X_test_tmp,np.power(X_test[:,t-m:t+m+1],i)]
                
            
        regr = Ridge(alpha=alpha_)
        regr.fit(X_train_tmp, Y_train_tmp)
        y_pred = regr.predict(X_test_tmp)
        y_pred_tr = regr.predict(X_train_tmp)
        loss_te += mean_squared_error(y_pred , Y_test_tmp)
        loss_tr += mean_squared_error(y_pred_tr , Y_train_tmp)
            
    return  loss_te/(200-(2*m)) ,loss_tr/(200-(2*m))

def cv_regularization(data,alphas,n):
    seed = 123
    k_fold = 4
    num_observations = 4100
    # split data in k fold
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    k_indices = build_k_indices_n(data, k_fold, seed) 
    
    for alpha in alphas:
        rmse_tr_tmp = 0
        rmse_te_tmp = 0
        
        for k in range(k_fold):
            X_train,X_test,Y_train,Y_test = prepare_data_xyw1(data,k_indices ,k)
            loss_test , loss_train = cross_validation_reg( X_train, X_test, Y_train, Y_test, alpha,n,1)
            rmse_te_tmp +=(loss_test)
            rmse_tr_tmp +=(loss_train)
        rmse_tr.append(rmse_tr_tmp/k_fold)
        rmse_te.append(rmse_te_tmp/k_fold) 
        
    return rmse_te ,rmse_tr
######################################### K_fold validation for Augmented inputs###########################################
def cv_regularization_degree(data,alphas,n,degree):
    seed = 123
    k_fold = 4
    num_observations = 4100
    # split data in k fold
    
    # define lists to store the loss of training data and test data
    rmse_tr_deg = []
    rmse_te_deg = []
    best_alpha =  []
    # cross validation
    k_indices = build_k_indices_n(data, k_fold, seed) 
    
    for degree_ in degree:
        
        rmse_tr = []
        rmse_te = []
        for alpha in alphas:
            rmse_tr_tmp = 0
            rmse_te_tmp = 0

            for k in range(k_fold):
                X_train,X_test,Y_train,Y_test = prepare_data_xyw1(data,k_indices ,k)
                loss_test , loss_train = cross_validation_reg( X_train, X_test, Y_train, Y_test, alpha,n,degree_)
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




##################### AVERAGE CP AND AVERAGE CL ##################################################
##################################################################################################
##################################################################################################



def prepare_data_average(data,k_indices ,k, degree = 1):
    Y = np.mean(data[:,1,:] , axis = 1)
    X = np.copy(data[:,0,:])
    if degree >=2 :
        for i in range(2,degree+1):
            X = np.c_[X, np.power(X,i)]
        
    num_observations = 4100
    index_test = k_indices[k]
    index_train = [ i for i in range(num_observations) if i not in k_indices[k] ] 
    X_train = X[index_train,:]
    X_test = X[index_test,:]
    Y_train = Y[index_train]
    Y_test = Y[index_test]
    return X_train,X_test,Y_train,Y_test

def average_cv_reg( X_train, X_test, Y_train, Y_test, alpha_ ):
    """return the loss of ridge regression."""
    # get k'th subgroup in test, others in train
    loss_te = 0
    loss_tr =0
    regr = Ridge(alpha=alpha_)
    regr.fit(X_train, Y_train)
    y_pred = regr.predict(X_test)
    y_pred_tr = regr.predict(X_train)
    loss_te = mean_squared_error(y_pred , Y_test)
    loss_tr = mean_squared_error(y_pred_tr , Y_train)
    return  loss_te ,loss_tr

def average_cv_regularization(data,alphas, degree = 1):
    seed = 123
    k_fold = 4
    num_observations = 4100
    # split data in k fold
    
    # define lists to store the loss of training data and test data
    rmse_tr = []
    rmse_te = []
    # cross validation
    k_indices = build_k_indices_n(data, k_fold, seed) 
    
    for alpha in alphas:
        rmse_tr_tmp = 0
        rmse_te_tmp = 0
        
        for k in range(k_fold):
            X_train,X_test,Y_train,Y_test = prepare_data_average(data,k_indices ,k, degree)
            loss_test , loss_train = average_cv_reg( X_train, X_test, Y_train, Y_test, alpha)
            rmse_te_tmp +=(loss_test)
            rmse_tr_tmp +=(loss_train)
        rmse_tr.append(rmse_tr_tmp/k_fold)
        rmse_te.append(rmse_te_tmp/k_fold) 
        
    return rmse_te ,rmse_tr
def simple_cv(data,degree=1):
    seed=123
    k_fold = 4
    mse_tr = 0
    mse_te = 0
    k_fold = 4
    # cross validation
    k_indices = build_k_indices_n(data, k_fold, seed)
    for k in range(k_fold):
            X_train,X_test,Y_train,Y_test = prepare_data_average(data,k_indices ,k,degree)
            regr = linear_model.LinearRegression()
            regr.fit(X_train, Y_train)
            y_pred = regr.predict(X_test)
            y_pred_tr = regr.predict(X_train)
            mse_te += mean_squared_error(y_pred , Y_test)
            mse_tr += mean_squared_error(y_pred_tr , Y_train)
    return mse_te/k_fold , mse_tr/k_fold

