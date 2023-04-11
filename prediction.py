
# Load packages
import os
import ast
import warnings
import numpy as np
import random
import joblib
import lightgbm as lgb
from sklearn.linear_model import ElasticNet
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Input, LeakyReLU, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm
import tensorflow.keras.backend as K
from functools import partial
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
    params_elnet = params_all['Elastic Net']
    
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
    
    # Train models: ML benchmarks
    elnet_mod = ElasticNet(**params_elnet)
    elnet_mod.fit(X_train, y_train)
    
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
        
        # Predictions standard ML        
        y_pred_elnet = np.expand_dims(elnet_mod.predict(X_test), axis=1)    
        y_pred_baseline = baseline_model.predict(X_test)
    
        # Train models: WDGRL
        training_ratio = 1
        gen_rate = 0.0005
        gradient_penalty_weight = 1
        was_rate = 0.0005
        wd_param = 1
    
        def wasserstein_loss(y_true, y_pred):
            return K.mean(y_true * y_pred)
    
        def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
            gradients = K.gradients(y_pred, averaged_samples)[0]
            gradients_sqr = K.square(gradients)
            gradients_sqr_sum = K.sum(gradients_sqr, axis=np.arange(1, len(gradients_sqr.shape)))
            gradient_l2_norm = K.sqrt(gradients_sqr_sum)
            gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
            return K.mean(gradient_penalty)
    
        def make_wasserstein(neurons_fe):
            model = Sequential()
            model.add(Dense(int(neurons_reg),  input_shape = (int(neurons_fe),)))
            model.add(LeakyReLU())
            model.add(Dense(1))
            return model
    
        class RandomWeightedAverage(Layer):
            def __init__(self, batch_size):
                super().__init__()
                self.batch_size = batch_size
    
            def call(self, inputs):
                weights = K.random_uniform((self.batch_size, 1))
                return (weights * inputs[0]) + ((1 - weights) * inputs[1])
        
        feature_extractor = feature_extraction(neurons_fe)
        regressor = regression(neurons_reg)
        inputs = Input(shape = input_shape)
        features = feature_extractor(inputs)
        outputs = regressor(features)
    
        reg_model = Model(inputs = inputs, outputs = outputs)
        reg_model.compile(optimizer = Adam(learning_rate, decay = learning_rate/decay_factor), loss = ["mse"], metrics = ["mse"])
    
        generator_input_for_wasserstein1 = Input(shape=input_shape)
        generator_input_for_wasserstein2 = Input(shape=input_shape)
    
        generated_samples_for_wasserstein1 = feature_extractor(generator_input_for_wasserstein1)
        generated_samples_for_wasserstein2 = feature_extractor(generator_input_for_wasserstein2)
    
        wasserstein = make_wasserstein(neurons_fe)
    
        wasserstein_output_from_generator1 = wasserstein(generated_samples_for_wasserstein1)
        wasserstein_output_from_generator2 = wasserstein(generated_samples_for_wasserstein2)
    
        regressor_output_from_generator1 = regressor(generated_samples_for_wasserstein1)
    
        averaged_samples = RandomWeightedAverage(batch_size)([generated_samples_for_wasserstein2, generated_samples_for_wasserstein1])
        averaged_samples_out = wasserstein(averaged_samples)
        partial_gp_loss = partial(gradient_penalty_loss, averaged_samples=averaged_samples, gradient_penalty_weight=gradient_penalty_weight)
        partial_gp_loss.__name__ = "gradient_penalty" 
    
        for layer in wasserstein.layers:
            layer.trainable = True
        wasserstein.trainable = True
    
        for layer in feature_extractor.layers:
            layer.trainable = False
        feature_extractor.trainable = False
    
        for layer in regressor.layers:
            layer.trainable = False
        regressor.trainable = False
    
        wasserstein_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1], outputs=[wasserstein_output_from_generator2, wasserstein_output_from_generator1, averaged_samples_out])
        wasserstein_model.compile(optimizer=Adam(was_rate, decay = was_rate/decay_factor), loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])
    
        for layer in wasserstein.layers:
            layer.trainable = False
        wasserstein.trainable = False
    
        for layer in feature_extractor.layers:
            layer.trainable = True
        feature_extractor.trainable = True
    
        for layer in regressor.layers:
            layer.trainable = True
        regressor.trainable = True
    
        generator_model = Model(inputs=[generator_input_for_wasserstein2, generator_input_for_wasserstein1], outputs=[regressor_output_from_generator1, wasserstein_output_from_generator2, wasserstein_output_from_generator1])
        generator_model.compile(optimizer=Adam(gen_rate, decay = gen_rate/decay_factor), loss=["mse", wasserstein_loss, wasserstein_loss], loss_weights = [1, wd_param, wd_param])
    
        positive_y = np.ones((batch_size, 1), dtype=np.float32)
        negative_y = -positive_y
        dummy_y = np.zeros((batch_size, 1), dtype=np.float32)
    
        n_source = X_train.shape[0] // batch_size
        n_target = X_test.shape[0] // batch_size
    
        for epoch in range(epochs):
            for i in range(steps):
                ind_source = np.random.randint(0, n_source)
                ind_target = np.random.randint(0, n_target)
                x_source_batch = X_train[ind_source * batch_size:(ind_source + 1) * batch_size]
                y_source_batch = y_train[ind_source * batch_size:(ind_source + 1) * batch_size]
                x_target_batch = X_test[ind_target * batch_size:(ind_target + 1) * batch_size]     
                wasserstein_model.train_on_batch([x_target_batch, x_source_batch], [positive_y, negative_y, dummy_y])     
                if i % training_ratio == 0:
                    generator_model.train_on_batch([x_target_batch, x_source_batch], [y_source_batch, negative_y, positive_y])                     
        y_pred_wdgrl = reg_model.predict(X_test)
    
        file_name = "theta_" + str(theta) + "_run_" + str(run) + "_" + nonlin + "_" + error + "_pred_results.npz"
        np.savez(file_name, y_true=y_test, elnet=y_pred_elnet, baseline=y_pred_baseline, 
                 adversarial=y_pred_wdgrl)
