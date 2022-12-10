# Author: Ruinan Ma
# Email: r7ma@ucsd.edu
# This file defines the network structure and the backpropagation algorithm.

from cmath import log
import numpy as np

import util

class Activation():

    def __init__(self, activation_type):

        if activation_type not in ["sigmoid", "tanh", "ReLU","output"]:   #output can be used for the final layer. Feel free to use/remove it
            raise NotImplementedError(f"{activation_type} is not implemented.")

        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

        # Avoid divide by 0
        self.epsilon = 1e-2

    def __call__(self, z):

        return self.forward(z)

    def forward(self, z):

        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
 
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def ReLU(self, x):
        return np.maximum(0, x)

    def output(self, x):
        assert x.shape[0] == 10
        C = np.max(x, axis=0) # the term C is used to avoid overflow by exponentiating a large number

        return np.exp(x - C) / np.sum(np.exp(x - C), axis=0)

    def grad_sigmoid(self,x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))


    def grad_tanh(self,x):
        return 1 - np.square(self.tanh(x))

    def grad_ReLU(self,x):
        return np.where(x > 0, 1, 0)

    def grad_output(self, x):
        return 1
        


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation, weightType, batch_size=128):

        np.random.seed(42)

        self.w = None
        
        self.x = None   
        self.a = None   
        self.z = None   
        
        self.in_units = in_units
        self.out_units = out_units

        self.batch_size = batch_size

        std = np.sqrt(2. / (out_units + in_units + 1))
        self.w = np.random.normal(loc=0., scale=std, size=[out_units, in_units + 1]) * 0.01
        self.dw = 0
        
        self.activation = activation   
        self.v = 0

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        self.x = util.append_bias(x) 
        assert self.x.shape[0] == self.in_units + 1
        self.a = np.dot(self.w, self.x)
        assert self.a.shape[0] == self.out_units
        self.z = self.activation(self.a)
        assert self.z.shape[0] == self.out_units
        return self.z

    def backward(self, deltaCur, learning_rate, momentum_gamma=0, regularization='L2', gradReqd=True):
        # batch size
        m = deltaCur.shape[1]
        assert deltaCur.shape[0] == self.out_units
        # Ruinan: is not the REAL delta in the backpropgation formula! 
        # Still need to do a element-wise multiplication!
        deltaPrev = (self.w.T @ deltaCur)
        assert deltaPrev.shape[0] == self.in_units + 1
        assert deltaPrev.shape[1] == m
        self.v = momentum_gamma * self.v + learning_rate * (deltaCur@(self.x.T))
        self.w += learning_rate * self.v
        self.dw = deltaCur@(self.x.T)
        
        return deltaPrev[:-1]
        
    def backward_hidden(self, deltaCur, learning_rate, momentum_gamma=0, regularization='L2', gradReqd=True):
        m = deltaCur.shape[1]
        assert deltaCur.shape[0] == self.out_units
        deltaPrev = (self.w.T @ (self.activation.backward(self.a) * deltaCur))
        assert deltaPrev.shape[0] == self.in_units + 1
        assert deltaPrev.shape[1] == m
        self.v = momentum_gamma * self.v + learning_rate * self.activation.backward(self.a) * deltaCur@(self.x.T)
        self.w += learning_rate * self.v 
        self.dw = self.activation.backward(self.a) * deltaCur@(self.x.T)
        return deltaPrev[:-1]
    
    

        


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.

    """

    def __init__(self, config):
        self.layers = [] 
        self.num_layers = len(config['layer_specs']) - 1  
        self.x = None 
        self.y = None       
        self.targets = None  

        self.learning_rate = config['learning_rate']  
        self.momentum_gamma = config['momentum_gamma']  
        self.L2_penalty = config['L2_penalty']  

        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(
                    Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation(config['activation']),
                          config["weight_type"]))
            elif i == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output"),
                                         config["weight_type"]))

    def __call__(self, x, targets=None):
        return self.forward(x, targets)


    def forward(self, x, targets=None):
        self.targets = targets
        for i in range(self.num_layers):
            x = self.layers[i].forward(x)
        
        self.y = x

        loss, acc = self.loss(self.y, targets)
        return loss, acc


    def loss(self, logits, targets):
        loss = -np.sum(targets * np.log(logits))
        acc = np.sum(np.argmax(logits, axis=0) == np.argmax(targets, axis=0))
        
        return loss, acc
    
    def backward(self, gradReqd=True):
        deltaCur =  self.targets - self.y
        deltaCur = self.layers[-1].backward(deltaCur=deltaCur, learning_rate=self.learning_rate, 
                                            momentum_gamma=self.momentum_gamma, gradReqd = True)
        for i in range(-2, -len(self.layers) - 1, -1):
            deltaCur = self.layers[i].backward_hidden(deltaCur=deltaCur, learning_rate=self.learning_rate, 
                                                        momentum_gamma=self.momentum_gamma, gradReqd = True)
