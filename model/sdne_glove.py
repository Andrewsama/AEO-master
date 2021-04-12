from tensorflow.contrib.layers import fully_connected

import numpy as np
import math
import tensorflow as tf
import time
import copy
import random
import os

from model.rbm import *

class SDNE:
    def __init__(self, config):
    
        self.is_variables_init = False
        self.config = config 
        ######### not running out gpu sources ##########
        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config =  tf_config)

        ############ define variables ##################
        self.layers = len(config.struct)
        self.struct = config.struct
        self.sparse_dot = config.sparse_dot
        self.W = {}
        self.b = {}
        self.l2_reg = 0.001
        self.MAX = tf.constant(100.0, name='X_max')
        struct = self.struct
        for i in range(self.layers - 1):
            name = "encoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        struct.reverse()
        for i in range(self.layers - 1):
            name = "decoder" + str(i)
            self.W[name] = tf.Variable(tf.random_normal([struct[i], struct[i+1]]), name = name)
            self.b[name] = tf.Variable(tf.zeros([struct[i+1]]), name = name)
        self.struct.reverse()
        ###############################################
        ############## define input ###################
                
        self.adjacent_matriX = tf.placeholder("float", [None, None]) #config.struct[-1]])
        # these variables are for sparse_dot
        self.X_sp_indices = tf.placeholder(tf.int64)
        self.X_sp_ids_val = tf.placeholder(tf.float32)
        self.X_sp_shape = tf.placeholder(tf.int64)
        self.X_sp = tf.SparseTensor(self.X_sp_indices, self.X_sp_ids_val, self.X_sp_shape)
        #
        #self.X = tf.placeholder("float", [None, config.struct[0]])

        self.COMATRIX = tf.placeholder("float", [None, config.struct[0]])
        
        ###############################################
        self.__make_compute_graph()
        self.loss = self.__make_loss(config)
        self.optimizer = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.loss)
        

    
    def __make_compute_graph(self):
        with tf.contrib.framework.arg_scope(
                [fully_connected],
                activation_fn = tf.nn.elu,
                weights_initializer = tf.contrib.layers.variance_scaling_initializer(),
                weights_regularizer = tf.contrib.layers.l2_regularizer(self.l2_reg)):

            struct = self.struct
            self.H = fully_connected(self.COMATRIX, struct[-1])
            #hidden2 = fully_connected(hidden1, n_hidden2)
            self.outputs = fully_connected(self.H, struct[0], activation_fn = None)

    def __make_loss(self, config):
        def get_loss_co(E, C, theta):
            return tf.reduce_sum(tf.pow(tf.div(C,self.MAX),theta) * (tf.pow(tf.tensordot(E, tf.transpose(E), axes=1)-tf.log1p(C), 2)))
        def func(X, theta):
            return tf.cond(X < self.MAX, lambda: tf.pow(tf.div(X,self.MAX),theta), lambda: 1)

        def get_1st_loss_link_sample(self, Y1, Y2):
            return tf.reduce_sum(tf.pow(Y1 - Y2, 2))
        def get_1st_loss(H, adj_mini_batch):
            D = tf.diag(tf.reduce_sum(adj_mini_batch,1))
            L = D - adj_mini_batch ## L is laplation-matriX
            return 2*tf.trace(tf.matmul(tf.matmul(tf.transpose(H),L),H))

        def get_co_loss(E, C, theta):
            return tf.reduce_sum(tf.square(theta*E - tf.log1p(C)))

        def get_2nd_loss(X, newX, beta):
            B = X * (beta - 1) + 1
            return tf.reduce_sum(tf.square(B * (newX - X)))

        def get_reg_loss(weight, biases):
            ret = tf.add_n([tf.nn.l2_loss(w) for w in weight.values()]) #itervalues()])
            ret = ret + tf.add_n([tf.nn.l2_loss(b) for b in biases.values()])
            return ret
            
        #Loss function
        self.loss_co = get_loss_co(self.H, self.COMATRIX, config.theta)
        self.co_cost = get_co_loss(self.H, self.adjacent_matriX, config.theta)

        self.reconstruction_loss = get_2nd_loss(self.COMATRIX, self.outputs, config.beta)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        return tf.add_n([self.reconstruction_loss, self.co_cost] + reg_losses)

    def save_model(self, path):
        bb = list(self.b.values())
        ww = list(self.W.values())
        saver = tf.train.Saver(bb + ww)
        saver.save(self.sess, path)

    def restore_model(self, path):
        saver = tf.train.Saver(list(self.b.values() ) + list(self.W.values() ))
        saver.restore(self.sess, path)
        self.is_Init = True
    
    def do_variables_init(self, data):
        def assign(a, b):
            op = a.assign(b)
            self.sess.run(op)
        init = tf.global_variables_initializer()       
        self.sess.run(init)
        if os.path.exists(self.config.restore_model):
            self.restore_model(self.config.restore_model)
            print("restore model" + self.config.restore_model)
        elif self.config.dbn_init:
            shape = self.struct
            myRBMs = []
            for i in range(len(shape) - 1):
                myRBM = rbm([shape[i], shape[i+1]], {"batch_size": self.config.dbn_batch_size, "learning_rate": self.config.dbn_learning_rate})
                myRBMs.append(myRBM)
                for epoch in range(self.config.dbn_epochs):
                    error = 0
                    for batch in range(0, data.N, self.config.dbn_batch_size):
                        mini_batch = data.sample(self.config.dbn_batch_size).COMATRIX
                        for k in range(len(myRBMs) - 1):
                            mini_batch = myRBMs[k].getH(mini_batch)
                        error += myRBM.fit(mini_batch)
                    print("rbm epochs:", epoch, "error : ", error)

                W, bv, bh = myRBM.getWb()
                name = "encoder" + str(i)
                assign(self.W[name], W)
                assign(self.b[name], bh)
                name = "decoder" + str(self.layers - i - 2)
                assign(self.W[name], W.transpose())
                assign(self.b[name], bv)
        self.is_Init = True

    def __get_feed_dict(self, data):
        X = data.X
        if self.sparse_dot:
            X_ind = np.vstack(np.where(X)).astype(np.int64).T
            X_shape = np.array(X.shape).astype(np.int64)
            X_val = X[np.where(X)]
            return {self.X : data.X, self.X_sp_indices: X_ind, self.X_sp_shape:X_shape, self.X_sp_ids_val: X_val, self.adjacent_matriX : data.adjacent_matriX}
        else:
            return {self.COMATRIX: data.COMATRIX, self.adjacent_matriX: data.adjacent_matriX}

    def fit(self, data):
        feed_dict = self.__get_feed_dict(data)
        ret, _ = self.sess.run((self.loss, self.optimizer), feed_dict = feed_dict)
        return ret
    
    def get_loss(self, data):
        feed_dict = self.__get_feed_dict(data)
        return self.sess.run(self.loss, feed_dict = feed_dict)

    def get_embedding(self, data):
        return self.sess.run(self.H, feed_dict = self.__get_feed_dict(data))

    def get_W(self):
        return self.sess.run(self.W)
        
    def get_B(self):
        return self.sess.run(self.b)
        
    def close(self):
        self.sess.close()
