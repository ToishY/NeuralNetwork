# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:05:18 2020

@author: ToshY
"""

import numpy as np

# Activation functions
class ActivationFunctions():
    """
    Providing several activation functions used at forward and 
    backward propogation in neural networks.

    Methods
    -------
    identity(x)
        Identity activation function
    identity_deriv(x)
        Derivative of Identity activation function
    sigmoid(x)
        Sigmoid activation function
    sigmoid_deriv(x)
        Derivative of Sigmoid activation function
    tanh(x)
        Tanh activation activation function
    tanh_deriv(x)
        Derivative of Tanh activation function
    arctan(x)
        Arctan activation activation function
    arctan_deriv(x)
        Derivative of Arctan activation function
    arcsin(x)
        Arcsin activation function
    arcsin_deriv(x)
        Derivative of Arcsin activation function
    elu(x, alpha=0.01)
        ELU activation function
    elu_deriv(x, alpha=0.01)
        Derivative of ELU activation function
    selu(x, alpha = 1.67326, lmbda = 1.0507)
        SELU activation function
    selu_deriv(x, alpha = 1.67326, lmbda = 1.0507)
        Derivative of SELU activation function
    relu(x)
        RELU activation function
    relu_deriv(x)
        Derivative of RELU activation function
    prelu(x, alpha=0.01)
        PRELU activation function
    prelu_deriv(x, alpha=0.01)
        Derivative of PRELU activation function
    isru(x, alpha=0.01)
        ISRU activation function
    isru_deriv(x, alpha=0.01)
        Derivative of ISRU activation function
    isrlu(x, alpha=0.01)
        ISRLU activation function
    isrlu_deriv(x, alpha=0.01)
        Derivative of ISRLU activation function
    softsign(x)
        Softsign activation function
    softsign_deriv(x)
        Derivative of Softsign activation function
    softplus(x)
        Softplus activation function
    softplus_deriv(x)
        Derivative of Softplus activation function
    softexp(x)
        Softexp activation function
    softexp_deriv(x)
        Derivative of Softexp activation function
    bent(x)
        Bent activation function
    bent_deriv(x)
        Derivative of Bent activation function
    """
    
    def identity(x):
        return x
    
    def identity_deriv(x):
        return 1

    def sigmoid(x):
        return (1/(1+np.exp(-x)))
    
    def sigmoid_deriv(x):
        xs = (1/(1+np.exp(-x)))
        return (xs*(1-xs))
    
    def tanh(x):
        return np.tanh(x)
    
    def tanh_deriv(x):
        return (1-np.power(np.tanh(x),2))

    def arctan(x):
        return np.arctan(x)
    
    def arctan_deriv(x):
        return (1/(np.power(x,2)+1))
    
    def arcsin(x):
        return np.arcsin(x)
    
    def arcsin_deriv(x):
        return (1/np.sqrt(np.power(x,2)+1))
    
    def elu(x, alpha=0.01):
        x[x<=0] = (alpha*(np.exp(x[x<=0])-1))
        return x
    
    def elu_deriv(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x<= 0] = ((alpha*(np.exp(x[x<=0])-1))+alpha)
        return dx
    
    def selu(x, alpha=1.67326, lmbda=1.0507):
        x[x<0] = (alpha*(np.exp(x[x<0])-1))
        return (lmbda*x)
    
    def selu_deriv(x, alpha=1.67326, lmbda=1.0507):
        dx = np.ones_like(x)
        dx[x<0] = (alpha*np.exp(x[x<0]))
        return (lmbda * dx)
    
    def relu(x):
        x[x<0] = 0
        return x
    
    def relu_deriv(x):
        dx = np.ones_like(x)
        dx[x<0] = 0
        return dx
    
    def prelu(x, alpha=0.01):
        x[x<0] = alpha*x[x<0]
        return x
    
    def prelu_deriv(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x<0] = alpha
        return dx
    
    def isru(x, alpha=0.01):
        return (x/np.sqrt(1+alpha*np.power(x,2)))
    
    def isru_deriv(x, alpha=0.01):
        return np.power((1/np.sqrt(1+alpha*np.power(x,2))),3)
    
    def isrlu(x, alpha=0.01):
        x[x<0] = (x[x<0]/np.sqrt(1+alpha*np.power(x[x<0],2)))
        return x
    
    def isrlu_deriv(x, alpha=0.01):
        dx = np.ones_like(x)
        dx[x<0] = np.power((1/np.sqrt(1+alpha*np.power(x[x<0],2))),3)
        return dx
    
    def softsign(x):
        return (x/(1+np.abs(x)))
    
    def softsign_deriv(x):
        return (1/np.power((1+np.abs(x)),2))
    
    def softplus(x):
        return np.log(1+np.exp(x))
    
    def softplus_deriv(x):
        return (1/(1+np.exp(-x)))
    
    def softexp(x, alpha=0.01):
        xt = x
        xt[alpha<0] = ((-np.log(1-alpha*(x[alpha<0]+alpha)))/alpha)
        xt[alpha>0] = (((np.exp(alpha*x[alpha>0])-1)/alpha)+alpha)
        return xt
    
    def softexp_deriv(x, alpha=0.01):
        dx=x
        dx[alpha<0] = (1/(1-alpha*(alpha+x[alpha<0])))
        dx[alpha>=0] = np.exp(alpha*x[alpha>=0])
        return dx
        
    def bent(x):
        return (((np.sqrt(np.power(x,2)+1)-1)/2)+x)
    
    def bent_deriv(x):
        return ((x/(2*np.sqrt(np.power(x,2)+1)))+1)
