
#%% Load packages

import os
import warnings
import numpy as np
import random
import joblib
import lightgbm as lgb
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.constraints import MaxNorm

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
warnings.filterwarnings("ignore")

# Load real-world data variables: mean_train, cov_train, mean_test, cov_test, f_hat
rw_data = np.load('data/rw_data.npz')
mean_train = rw_data['mean_train']
cov_train = rw_data['cov_train']
mean_test = rw_data['mean_test']
cov_test = rw_data['cov_test']
diff_vec = mean_test - mean_train
sig_y = rw_data['sig_y']

nonlin = 'rf'
if nonlin == 'rf':
    f_hat = joblib.load("data/rf_model.pickle")
elif nonlin == 'gbm':
    f_hat = lgb.Booster(model_file='data/lgb_model.txt')
    

#%% Data generation

# Generate training data
N = 5830
gauss_mv_samples = np.random.multivariate_normal(mean_train, cov_train, size=N)
x_1 = gauss_mv_samples[:,0:1]
x_2 = gauss_mv_samples[:,1:2]
x_3 = gauss_mv_samples[:,2:3]
x_4 = gauss_mv_samples[:,3:4]
x_5 = gauss_mv_samples[:,4:5]
x_6 = gauss_mv_samples[:,5:6]
x_7 = gauss_mv_samples[:,6:7]
x_8 = gauss_mv_samples[:,7:8]
x_9 = gauss_mv_samples[:,8:9]
x_10 = gauss_mv_samples[:,9:10]
x_11 = gauss_mv_samples[:,10:11]
x_12 = gauss_mv_samples[:,11:12]
x_13 = gauss_mv_samples[:,12:13]
x_14 = gauss_mv_samples[:,13:14]
x_15 = gauss_mv_samples[:,14:15]
x_16 = gauss_mv_samples[:,15:16]
x_17 = gauss_mv_samples[:,16:17]
x_18 = gauss_mv_samples[:,17:18]
x_19 = gauss_mv_samples[:,18:19]
x_20 = gauss_mv_samples[:,19:20]
X_train = np.concatenate([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20], axis = 1)

error = 'gaussian'
if error == 'gaussian':
    y_train = f_hat.predict(X_train).reshape((N,1)) + np.random.normal(loc = 0, scale = sig_y, size = (N,1))
elif error == 'uniform':
    y_train = f_hat.predict(X_train).reshape((N,1)) + np.random.uniform(low = -(np.sqrt(12)*sig_y)/2, high = (np.sqrt(12)*sig_y)/2, size = (N,1))

# Cross validation 
cv = KFold(n_splits = 5, shuffle = True)

# Elastic Net
pipeline = ElasticNet()
params = {"alpha":[0.01, 0.1, 1.0, 10.0, 100.0],
          "l1_ratio":[0, 0.25, 0.5, 0.75, 1]}
model_cv_elnet = GridSearchCV(pipeline, param_grid = params, cv=cv, scoring = "neg_mean_absolute_error", verbose=0)  
model_cv_elnet.fit(X_train, y_train)
params_elnet = model_cv_elnet.best_params_

# ANN
input_shape = (X_train.shape[1],)
neurons_reg = 8
learning_rate = 0.0005
decay_factor = 300

def create_model(dropout, neurons_fe, regularization):
    model_ann = Sequential()
    model_ann.add(Dense(neurons_fe, input_shape = input_shape, kernel_regularizer = l2(regularization), 
                        activation = "relu", kernel_constraint=MaxNorm(3)))
    model_ann.add(Dropout(dropout))
    model_ann.add(Dense(neurons_reg, activation = "relu", kernel_constraint=MaxNorm(3)))
    model_ann.add(Dropout(dropout))
    model_ann.add(Dense(1, activation ="linear"))
    model_ann.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), 
                      loss=["mse"], metrics = ["mse"])
    return model_ann

pipeline = KerasRegressor(build_fn = create_model, verbose = 0)
params = {"epochs":[50, 100],                   
          "batch_size":[32, 64],
          "dropout":[0.4, 0.5, 0.6],             
          "regularization":[1e-2, 1e-3, 1e-5],
          "neurons_fe": [8, 16, 32, 64]}
model_cv_ann = GridSearchCV(pipeline, param_grid = params, cv=cv, scoring = "neg_mean_absolute_error", verbose = 0)  
model_cv_ann.fit(X_train, y_train)
params_ann = model_cv_ann.best_params_

# Save parameters
params_all = {
    'Elastic Net': params_elnet,
    'Artificial Neural Network': params_ann}

file_name = "Optim_hyperpar_" + nonlin + "_" + error + ".txt"
file = open(file_name, "w")
file.write(repr(params_all))
file.close()  
