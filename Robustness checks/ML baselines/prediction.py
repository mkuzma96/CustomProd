
# Load packages

import os
import ast
import warnings
import numpy as np
import random
import joblib
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()
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
    
#%% Generate train and test data

# Seeds are set from 1,...,10
for run in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
    random.seed(run)
    np.random.seed(run)
    tf.random.set_seed(run)
    
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
    
    # Load hyperparameters
    hyper_path = "data/" + "Optim_hyperpar_" + nonlin + "_" + error + ".txt"
    hyperpars = open(hyper_path,'r').read()
    params_all = ast.literal_eval(hyperpars)
    
    input_shape = (X_train.shape[1],)
    neurons_reg = 8
    learning_rate = 0.0005
    decay_factor = 300
    epochs = int(params_all['Artificial Neural Network']['epochs'])
    batch_size = int(params_all['Artificial Neural Network']['batch_size'])
    neurons_fe = int(params_all['Artificial Neural Network']['neurons_fe'])
    regularization = params_all['Artificial Neural Network']['regularization']
    dropout = params_all['Artificial Neural Network']['dropout']
    steps = int(X_train.shape[0]/batch_size)
       
    def feature_extraction(neurons_fe):
        model = Sequential()
        model.add(Dense(neurons_fe, input_shape = input_shape, kernel_regularizer = l2(regularization),
                        activation = "relu", kernel_constraint=MaxNorm(3)))
        model.add(Dropout(dropout))
        return model
    
    def regression(neurons_reg):
        model = Sequential()
        model.add(Dense(neurons_reg, activation = "relu", kernel_constraint=MaxNorm(3)))
        model.add(Dropout(dropout))
        model.add(Dense(1, activation ="linear"))
        return model

    feature_extractor = feature_extraction(neurons_fe)
    regressor = regression(neurons_reg)
    inputs = Input(shape = input_shape)
    features = feature_extractor(inputs)
    outputs = regressor(features)   
    
    steps = int(X_train.shape[0]/batch_size)
    n_source = X_train.shape[0] // batch_size
    
    baseline_model = Model(inputs=inputs, outputs=outputs)
    baseline_model.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), 
                           loss=["mse"], metrics = ["mse"])
    for epoch in range(epochs):
        for i in range(steps):
            ind_source = np.random.randint(0, n_source)
            x_source_batch = X_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            y_source_batch = y_train[ind_source * batch_size:(ind_source + 1) * batch_size]
            baseline_model.train_on_batch(x_source_batch, y_source_batch) 
                
    # Generate test data with parameter theta = [1, 2, 3, 4] controling the covariate shift
    test_size = 3866
    thetas = [1, 2, 3, 4]
    for theta in thetas:
        gauss_mv_samples = np.random.multivariate_normal(mean_train + theta*diff_vec, cov_test, size=test_size)
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
        X_test = np.concatenate([x_1, x_2, x_3, x_4, x_5, x_6, x_7, x_8, x_9, x_10, x_11, x_12, x_13, x_14, x_15, x_16, x_17, x_18, x_19, x_20], axis = 1)
        
        if error == 'gaussian':
            y_test = f_hat.predict(X_test).reshape((test_size,1)) + np.random.normal(loc = 0, scale = sig_y, size = (test_size,1))
        elif error == 'uniform':
            y_test = f_hat.predict(X_test).reshape((test_size,1)) + np.random.uniform(low = -(np.sqrt(12)*sig_y)/2, high = (np.sqrt(12)*sig_y)/2, size = (test_size,1))
        
        days = 30
        idx1 = y_test <= days
        idx2 = y_test > days
        idx1 = idx1.reshape(-1)
        idx2 = idx2.reshape(-1)        
        y_test_new = y_test[idx2,:].copy()
        
        # Model retraining        
        y_train_retrain = np.concatenate([y_train.copy(), y_test[idx1,:].copy()], axis=0)
        X_train_retrain = np.concatenate([X_train.copy(), X_test[idx1,:].copy()], axis=0)
        
        feature_extractor = feature_extraction(neurons_fe)
        regressor = regression(neurons_reg)
        inputs = Input(shape = input_shape)
        features = feature_extractor(inputs)
        outputs = regressor(features)   
        
        steps = int(X_train_retrain.shape[0]/batch_size)
        n_source = X_train_retrain.shape[0] // batch_size
        
        baseline_model_retrain = Model(inputs=inputs, outputs=outputs)
        baseline_model_retrain.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), 
                                loss=["mse"], metrics = ["mse"])
        for epoch in range(epochs):
            for i in range(steps):
                ind_source = np.random.randint(0, n_source)
                x_source_batch = X_train_retrain[ind_source * batch_size:(ind_source + 1) * batch_size]
                y_source_batch = y_train_retrain[ind_source * batch_size:(ind_source + 1) * batch_size]
                baseline_model_retrain.train_on_batch(x_source_batch, y_source_batch)
        y_pred_retrain = baseline_model_retrain.predict(X_test[idx2,:])
          
        # Transfer learning
        y_train_transfer = y_test[idx1,:].copy()
        X_train_transfer = X_test[idx1,:].copy()
        
        steps = int(X_train_transfer.shape[0]/batch_size)
        n_source = X_train_transfer.shape[0] // batch_size
        
        baseline_model_transfer = tf.keras.models.clone_model(baseline_model)
        baseline_model_transfer.build((None,input_shape))
        baseline_model_transfer.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), 
                                loss=["mse"], metrics = ["mse"])
        baseline_model_transfer.set_weights(baseline_model.get_weights())
        
        for epoch in range(epochs):
            for i in range(steps):
                ind_source = np.random.randint(0, n_source)
                x_source_batch = X_train_transfer[ind_source * batch_size:(ind_source + 1) * batch_size]
                y_source_batch = y_train_transfer[ind_source * batch_size:(ind_source + 1) * batch_size]
                baseline_model_transfer.train_on_batch(x_source_batch, y_source_batch)
        y_pred_transfer = baseline_model_transfer.predict(X_test[idx2,:])
        
        file_name = "theta_" + str(theta) + "_run_" + str(run) + "_" + nonlin + "_" + error + "_pred_results.npz"
        np.savez(file_name, index=idx2, y_true=y_test_new, retrain=y_pred_retrain, transfer=y_pred_transfer)
