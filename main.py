# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:06:52 2020
Modified on Sun May 10 12:28:00 2020

@author: ToishY

This is an original work derived from previous work and a wide range of snippets,
with the purpose of creating a versatile Multilayer Perceptron.

* SOURCES (SNIPPETS+INFO)
@Ritchie Vink https://github.com/ritchie46/vanilla-machine-learning
@Michael Nielsen https://github.com/mnielsen/neural-networks-and-deep-learning
@Valerio Velardo https://github.com/musikalkemist/DeepLearningForAudioWithPython
@ML-Cheatsheet https://ml-cheatsheet.readthedocs.io/
@Mxnet https://gluon.mxnet.io

* REMARKS
The explanations from @Michael and YouTube tutorials of @Valerio (YT: TheSoundOfAI) 
really helped with understanding the basics of Neural Networks. While I initially 
followed along with @Valerio's tutorials, in the end I switched to use @Ritchie's 
snippet as base for this script, because I liked the use of dictonaries and the 
implementation of batches. The optimizers, currently Momentum and Adam, were added
a bit later. 

For me the most difficult to actually understand was the idea of backpropagation.
So I followed @Valerio's explanation regarding backpropagation
(https://www.youtube.com/watch?v=ScL18goxsSg&list=PL-wATfeyAMNrtbkCNsLcpoAyBBRJZVlnf).
 
I took my whiteboard out and created a random 4 layer network, with 2 hidden layers, 
and just started writing out the backpropagation steps. If you really take the time 
to write it the derivatives, and follow along, you realize that it's called backprop because
you start with calculating ∂L/dW_i from the last layer, and after that you do the same for
the second to last layer (∂L/dW_i-1). And what you'll notice is that the expression 
for second to last layer contains elements which were already calculated from the
previous layer. Below a short example with the derivations for a 4 layered network.

===========================================================

The idea in general:

 ∂L        ∂L     ∂a_i+1   ∂z_i+1 
----  =  -----    -----    ------
∂W_i     ∂a_i+1   ∂z_i+1    ∂W_i

Where ∂E/∂a_i+1

 ∂L       ∂L      ∂a_i+2   ∂z_i+2 
----  =  -----    -----   ------
∂a_i+1   ∂a_i+2   ∂z_i+2   ∂a_i+1


===========================================================

EXAMPLE: 4 layers, 2 hidden layers

* = node; a/z

1     2    3    4

           * 
      *         
* w1    w2 * w3 *
      *   
           *

So for W3 it's easy:

 ∂L       ∂L     ∂a_4    ∂z_4 
----  =  -----  -----   ------
∂W_3     ∂a_4    ∂z_4    ∂W_3
       
Now doing the same for W2:
    
 ∂L       ∂L     ∂a_3    ∂z_3 
----  =  -----  -----   ------
∂W_2     ∂a_3    ∂z_3    ∂W_2

We know everything except ∂L/∂a_3. So by using the chain rule again:

 ∂L        ∂L    ∂a_4    ∂z_4
----  =  -----   -----  ------
∂a_3      ∂a_4   ∂z_4    ∂a_3

Substituting this back in:

 ∂L      /  ∂L    ∂a_4    ∂z_4  \   ∂a_3    ∂z_3 
----  = | -----  -----  ------  |  -----   ------
∂W_2    \ ∂a_4   ∂z_4    ∂a_3  /    ∂z_3    ∂W_2

Now we see that the elements...
    
 ∂L    ∂a_4
-----  -----
∂a_4   ∂z_4

...were already calculated in the expresion for ∂L/∂W_3.

Now you can do the the same process for ∂L/∂W_1, which will give you:
    
 ∂L      //  ∂L    ∂a_4    ∂z_4  \   ∂a_3    ∂z_3  \   ∂a_2    ∂z_2
----  = || -----  -----   -----  |  -----   ------ |  -----   ------
∂W_w    \\ ∂a_4   ∂z_4    ∂a_3  /    ∂z_3    ∂a_2 /    ∂z_2    ∂W_1

Now we see that the elements...
    
  ∂L    ∂a_4   ∂z_4   ∂a_3
-----  -----  -----  -----
∂a_4   ∂z_4   ∂a_3   ∂z_3

...were already calculated in the expresion for ∂L/∂W_2.

So because you've calculated the elements in the "previous" iteration, 
it makes the calculation for the current layer a lot easier.

===========================================================

* NOTES
- Tested with binary and multiclass sklearn and UCI datasets.
- Compared this Network v.s. Keras with sklearn moons dataset, 
  resulting in similar accuracy.

* TODO
- Add more gradient descent optimizer functions
- Use K-Fold for automatically determining amount of neurons in hidden layer(s)
- Maybe better checking of arguments parsed. Loading/saving model
- (Web)GUI?
        
"""
# %% 
import time
import os
import inspect
import collections
import numpy as np
from pathlib import Path
from src.metrics import Metrics

# Current working directory
cwd = str(Path(__file__).parent.absolute())

# Loss functions
class Quadratic(object):
    """ Mean Squared """ 
    
    @staticmethod
    def fn(y, yhat):
        return np.mean((y - yhat)**2)
    
    @staticmethod
    def dfn(y, yhat, z):
        return (yhat-y)*z
    
class Root(object):
    """ Root Mean Squared """
    
    @staticmethod
    def fn(y, yhat):
        return np.sqrt(np.mean((y - yhat)**2))

    @staticmethod
    def dfn(y, yhat, z):
        return (1/(2*np.sqrt(np.mean((y - yhat)**2))))*z

class Log(object):
    """ Cross Entropy  """

    @staticmethod
    def fn(y, yhat):
        return np.sum(np.nan_to_num(-y*np.log(yhat)-(1-y)*np.log(1-yhat)))

    @staticmethod
    def dfn(y, yhat, z):
        return (yhat-y)

# Neural network  
class NeuralNetwork():
    """Set up Neural Network model"""
    
    def __init__(self, X, y, y_class, btm=True, hidden=[3], activation_functions=['relu','sigmoid'], weight_initialisation=['he_normal','xavier_uniform'], batches=32, learning_rate=1e-3, cost=Quadratic, optimizer='GD', optimizer_options={}, verbose=True, seed=666):
        # Input data, separated by features and targets
        self.X = X
        self.y = y
        
        # Classes
        self.yclass = y_class
        self.btm = btm
        
        # Set seed
        self.seed = seed
        
        # Activation functions per layer and check if valid
        for a in activation_functions:
            self._check_activation_function( a )
        
        print('\n'+('Model settings').center(50,'-')+'\n')
        # Check if minibatches is specified
        if isinstance(batches, int) and batches >= 1:
            self.batch_mode=True
            self.batch_size=batches
            print(">> Batch size set to: {} samples".format(self.batch_size))
        else:
            self.batch_mode=False
            self.batch_size=len(self.X)
            print(">> Batch size set to: entire training set")
            
        # Learning rate
        self.lr = learning_rate
        
        # Loss function
        self.loss = cost
        self.loss_str = str(self.loss.__name__) + ' loss'
        
        # Check optimizer and options
        self.optvals = self._check_optimizer(optimizer, optimizer_options)
        self.optfn = getattr(self, optimizer)
        
        # Cross Validation
        self.cv_fn = None
        
        # The combined layers with amount of neurons for each one
        self.layers = [self.X.shape[1]] + hidden + [self.y.shape[1]]
        self.ll = len(self.layers)
        
        # Weight initialisation methods
        if len(weight_initialisation) != (self.ll-1):
            raise Exception('Only {} out of {} weight initialisation methods specified.'.format(len(weight_initialisation), (self.ll-1)))
            
        self.wi = weight_initialisation
        
        # Set activation functions
        self.actf = self._set_activations(activation_functions)
        
        # Verbose?
        self.verbose = verbose
        
    def _set_weight(self, weight_init=[]):
        """ Initialize random weights """
            
        wd = {}
        for idx in range(self.ll-1):
            for method in self.wi:
                wd[idx+1] = self._weight_method(idx, method)
                
        return wd
        
    def _weight_method(self, idx, method='uniform'):
        """ Weight initialisation methods 
        
        Uniform - Tanh
        Xavier - Sigmoid
        He - (P)Relu
        
        """
        
        # Set seed
        np.random.seed(self.seed)
        
        if method == 'uniform':
            return np.random.randn(self.layers[idx], self.layers[idx+1])*np.sqrt(1/self.layers[idx])
        elif method == 'xavier_normal':
            return np.random.randn(self.layers[idx], self.layers[idx+1])*np.sqrt(2/(self.layers[idx]+self.layers[idx+1]))
        elif method == 'xavier_uniform':
            return np.random.randn(self.layers[idx], self.layers[idx+1])*np.sqrt(6/(self.layers[idx]+self.layers[idx+1]))
        elif method == 'he_normal':
            return np.random.randn(self.layers[idx], self.layers[idx+1])*np.sqrt(2/self.layers[idx])
        elif method == 'he_uniform':
            return np.random.randn(self.layers[idx], self.layers[idx+1])*np.sqrt(6/self.layers[idx])
        elif method == 'zeros':
            return np.zeros((self.layers[idx], self.layers[idx+1]))
        else:
            return np.random.randn(self.layers[idx], self.layers[idx+1])
                     
    def _set_bias(self, bias=0.0):
        """Initialize (zero) bias"""
        
        return {i+1: np.full(self.layers[i+1],bias) for i in range(self.ll-1)}
    
    def _set_activations(self, activation_functions):
        """ Initialize activations"""

        return {i+2: activation_functions[i] for i in range(self.ll-1)}
    
    def _set_zero_array(self):
        """ Initialize list of zero arrays; Used at activations and optimizer """
        
        return [np.zeros((x,y)) for x, y in zip(self.layers[:-1], self.layers[1:])]
    
    def _reshape_array(self, mat):
        return mat.reshape( mat.shape[0], -1 )
    
    def _check_optimizer(self, optimizer, options):
        # Optimizer template
        optimizer_template = {'GD':{'epoch':1},
                              'Momentum':{'epoch':1,'gamma':0.9},
                              'Adam':{'epoch':1,'beta_1':0.9, 'beta_2':0.999, 'eps':1e-8}}
        
        # Check if valid optimizer
        if optimizer not in optimizer_template:
            raise Exception('Invalid optimizer specified. Allowed: `{}`.'.format('`, `'.join(list(optimizer_template.keys()))))
        
        optr={'epoch':1}
        # Check Momentum options
        if optimizer == 'Momentum':
            if 'gamma' not in options:
                print('>> GD Momentum: `gamma` was set to default value: {}'.format(optimizer_template['Momentum']['gamma']))
                optr['gamma'] = 0.9
            else:
                optr['gamma'] = options['gamma']
        
        # Check Adam options
        if optimizer == 'Adam':
            if 'beta_1' not in options:
                print('>> GD Adam: `beta_1` was set to default value: {}'.format(optimizer_template['Adam']['beta_1']))
                optr['beta_1'] = 0.9
            else:
                print('>> GD Adam: `beta_1` was set to custom user value: {}'.format(options['beta_1']))
                optr['beta_1'] = options['beta_1']
            
            if 'beta_2' not in options:
                print('>> GD Adam: `beta_2` was set to default value: {}'.format(optimizer_template['Adam']['beta_2']))
                optr['beta_2'] = 0.999
            else:
                print('>> GD Adam: `beta_2` was set to custom user value: {}'.format(options['beta_2']))
                optr['beta_2'] = options['beta_2']
                
            if 'eps' not in options:
                print('>> GD Adam: `eps` was set to default value: {}'.format(optimizer_template['Adam']['eps']))
                optr['eps'] = 1e-8
            else:
                print('>> GD Adam: `eps` was set to custom user value: {}'.format(options['eps']))
                optr['eps'] = options['eps']
    
        return optr
        
    def _check_crossval_function(self, crossval, options):
        """ Check crosvalidation options """
        
        # Crossvalidation template
        crossval_template = {'k_fold':{'folds':10,'seed':self.seed,'shuffle':True},
                              'k_fold_strat':{'folds':10,'seed':self.seed,'shuffle':True},
                              'loo':{}}
        
        # Check if valid crossvalidation function
        if crossval not in crossval_template:
            raise Exception('Invalid optimizer specified. Allowed: `{}`.'.format('`, `'.join(list(crossval_template.keys()))))
            
        optr = {}
        
        # Check LOOCV
        if crossval == 'loo':
            return optr
        
        # Left with K-Fold and Stratified K-Fold CV
        if 'folds' not in options:
            print('\n>> CV: `folds` was set to default value: {}'.format(crossval_template[crossval]['folds']))
            optr['folds'] = crossval_template[crossval]['folds']
        else:
            print('\n>> CV: `folds` was set to custom user value: {}'.format(options['folds']))
            optr['folds'] = options['folds']
        
        if 'random_state' not in options:
            print('>> CV: `seed` was set to default value: {}'.format(crossval_template[crossval]['seed']))
            optr['seed'] = crossval_template[crossval]['seed']
        else:
            print('>> CV: `seed` was set to custom user value: {}'.format(options['seed']))
            optr['seed'] = options['seed']
            
        if 'shuffle' not in options:
            print('>> CV: `shuffle` was set to default value: {}'.format(crossval_template[crossval]['shuffle']))
            optr['shuffle'] = crossval_template[crossval]['shuffle']
        else:
            print('>> CV: `eps` was set to custom user value: {}'.format(options['eps']))
            optr['shuffle'] = options['shuffle']
    
        return optr
        
    def _check_activation_function(self, activation_function):
        """ Check if valid activation function is passed"""
        
        if not hasattr(afc, activation_function):
            raise Exception('Invalid activation function `{}`'.format(activation_function))
    
    def _check_activation_function_args(self, activation_function, activation_function_args):
        """ Check if valid activation function arguments are passed"""
        
        function_args = inspect.getfullargspec(getattr(afc, 'activation_function'))[0]
        compare = lambda x, y: collections.Counter(x) == collections.Counter(y)
        if not compare(function_args, activation_function_args):
            raise Exception('Invalid activation function arguments. Valid = {}'.format(function_args))
    
    def _create_batches(self, X, y):
        """ Get batches """
        
        batches = []
        for i in range(0, X.shape[0], self.batch_size):
            batches.append((X[i:i + self.batch_size], y[i:i + self.batch_size]))
            
        return batches
    
    def forward_propagate(self, X_data):
        """Forward propagation"""
               
        # First layer is X_data w/o activation function
        z={}
        activation = {1: X_data}

        for idx in range(1, self.ll):
            # z_i+1 = X_i*w_i + B_i
            z[idx+1] = np.dot(activation[idx], self.weights[idx]) + self.bias[idx]
            # a = actf( z )
            activation[idx+1] = getattr(afc, self.actf[idx+1])(z[idx+1])
            
        return z, activation
            

    def back_propagate(self, y, y_hat, z):
        """ Back propagation """
        
        # ∂L/∂W_i = (∂L / ∂a_i+1) * (∂a_i+1 / ∂z_i+1)
        if self.loss == Quadratic or self.loss == Root:
            delta = self.loss.dfn(y, y_hat[self.ll], getattr(afc, self.actf[self.ll-1] + '_deriv')( y_hat[self.ll] ))
        elif self.loss == Log:
            delta = self.loss.dfn(y,y_hat[self.ll],0)
        
        # dW 
        dW = np.dot(y_hat[self.ll - 1].T, delta)
        # Dict for storage purpose
        params = { self.ll - 1: (dW, delta) }
            
        # Other (hidden) layers
        for idx in reversed(range(2, self.ll)):
            delta = np.dot(delta, self.weights[idx].T) * getattr(afc, self.actf[idx] + '_deriv')( z[idx] )
            dw = np.dot(y_hat[idx-1].T, delta)
            params[idx-1] = (dw, delta)
        
        # (Mini)batch gradient descent
        self.optfn(params, self.optvals)
            
    def GD(self, params, opts):
        """ Gradient Descent """
        
        for k, (dw, db) in params.items():
            db = np.mean(db, 0)
            # Update weights
            self.weights[k] -= self.lr * dw
            self.bias[k] -= self.lr * db
            
    def Momentum(self, params, opts):
        """ Gradient Descent with Momentum """
        
        for k, (dw, db) in params.items():   
            db = np.mean(db, 0)
            # Velocities
            self.v_w[k] = opts['gamma'] * self.v_w[k] + self.lr * dw
            self.v_b[k] = opts['gamma'] * self.v_b[k] + self.lr * db
              
            # Update weights
            self.weights[k] -= self.v_w[k]
            self.bias[k] -= self.v_b[k]
        
    def Adam(self, params, opts):
        """ Adam is a SGD method that computes individual adaptive 
        learning rates  for different parameters from estimates of 
        first and second-order moments of the gradients """
        
        v_w_adj = {}
        v_b_adj = {}
        s_w_adj = {}
        s_b_adj = {}
        for k, (dw, db) in params.items():
            db = np.mean(db, 0)
            # Moving average of the gradients
            self.v_w[k] = opts['beta_1']*self.v_w[k] + (1-opts['beta_2'])*dw
            self.v_b[k] = opts['beta_1']*self.v_b[k] + (1-opts['beta_2'])*db
            
            # Moving average of the squared gradients
            self.s_w[k] = opts['beta_2']*self.s_w[k] + (1-opts['beta_2'])*dw**2
            self.s_b[k] = opts['beta_2']*self.s_b[k] + (1-opts['beta_2'])*db**2
            
            # Compute bias-corrected first moment estimate
            v_w_adj[k] = self.v_w[k]/(1-opts['beta_1']**opts['epoch'])
            v_b_adj[k] = self.v_b[k]/(1-opts['beta_1']**opts['epoch'])

            # Compute bias-corrected second moment estimate
            s_w_adj[k] = self.s_w[k]/(1-opts['beta_2']**opts['epoch'])
            s_b_adj[k] = self.s_b[k]/(1-opts['beta_2']**opts['epoch'])
    
            # Update weights
            self.weights[k] -= self.lr * (v_w_adj[k]/np.sqrt(s_w_adj[k]+opts['eps']))
            self.bias[k] -= self.lr * (v_b_adj[k]/np.sqrt(s_b_adj[k]+opts['eps']))
        
    def training(self, epochs, cross_val={}, method_opt='', onehotenc=object, test_metric=Root, normalize=True):
        """ Check for normal training or with CV """
        
        Xt = self.X
        yt = self.y
        
        # Test metric (for CV)
        self.fnt = test_metric
        
        # Normalize predictions between 0 and 1
        self.norm_pred = normalize
        
        # Timer
        st = time.time()
        
        if method_opt == 'single':
            print('\n'+(' Start Training ').center(50,'-')+'\n')
            if not self.verbose: print('...')
            training_results = self.train(Xt, yt, epochs)
        elif method_opt == 'multi':
            cvf = list(cross_val.keys())[0]
            cv_options = self._check_crossval_function(cvf, cross_val[cvf])
            self.cv_fn = cvf
            print('\n'+(' Start CV {} Training '.format(cvf.upper())).center(50,'-')+'\n')
            if not self.verbose: print('...')
            training_results = self.train_cv(Xt, yt, epochs, cv_options, onehotenc)
        else:
            raise Exception('Invalid method provided. Should contain key `split`, `k_fold` or `loo`.')
        
        ed = time.time()
        print('\n'+(' Training complete ').center(50,'-')+'\n')
        print('Elapsed time = {:.2f} seconds'.format(ed-st))
        return training_results
        
    def train(self, Xt, yt, epochs):
        """ Train model """
        
        # Due to addition of CV, weight initialisation now done here 
        
        # Set weigths & bias
        self.weights = self._set_weight()
        self.bias = self._set_bias()
        
        # Momentum: velocities
        if self.optfn.__name__ in ['Momentum', 'Adam']:
            self.v_w = {i: np.zeros(self.weights[i].shape) for i in range(1,self.ll)}
            self.v_b = {i: np.zeros(self.bias[i].shape) for i in range(1,self.ll)}
        
        # Adam: Exponential weighted average of the squared gradient
        if self.optfn.__name__ in ['Adam']:
            self.s_w = {i: np.zeros(self.weights[i].shape) for i in range(1,self.ll)}
            self.s_b = {i: np.zeros(self.bias[i].shape) for i in range(1,self.ll)}
        
        mse = []
        for e in range(1, epochs+1):
            sum_errors = 0
                
            # Get batch(es)
            X_train = self._create_batches(Xt, yt)
                
            # Iterate through data
            for idx, (sample, target) in enumerate(X_train):
                    
                # Get random minibatch if specified
                if self.batch_mode:
                    idx = np.random.randint(0, len(X_train))
                    sample, target = X_train[idx]
                
                # Forward
                z, output = self.forward_propagate(sample)
                
                # Backward
                self.back_propagate(target, output, z)
                
                # Append loss function result for later purposes
                sum_errors += self.loss.fn(target, output[self.ll])
            
                # Fix adding sum_errors per batch to dict!!!
                #se[iter] = sum_errors
            
            # Increment epoch (for Adam)
            self.optvals['epoch'] += 1
        
            # Append every 10 epochs to list
            if (e % 100) == 0:
                mse.append((e,sum_errors / len(sample)))
            
            # Display ever 100 epoch MSE
            if (e % 1000) == 0 and self.verbose:
                print("* EPOCH {}; ERROR = {}".format(e, sum_errors / len(sample)))
                
        return {'loss':{'train':mse},'weights':{'split':self.weights},'bias':{'split':self.bias}}

    def train_cv(self, Xt, yt, epochs, opts, ohe=None):
        """ Training with CV method """

        if self.cv_fn == 'k_fold_strat':
            # Transform to 1D to work with k-fold strat
            yt = ohe.inverse_transform(yt)
            cv = cvs(Xt, yt)
            splits, [X_train, y_train, X_val, y_val] = getattr(cv, self.cv_fn)(opts)
            # Reverse back by transforming with earlier defined OneHotEncoder
            for yv in enumerate(y_train):
                y_train[yv[0]] = ohe.transform(y_train[yv[0]]).toarray()
                y_val[yv[0]] = ohe.transform(y_val[yv[0]]).toarray()      
        else:
            cv = cvs(Xt, yt)
            splits, [X_train, y_train, X_val, y_val] = getattr(cv, self.cv_fn)(opts)
            
        train_fmse = []
        val_fmse = []
        val_pred = []
        val_true = []
        fold_met = []
        
        fold_weights = {}
        fold_bias = {}
        for f in range(splits):
            print('\nFold: {}\n'.format(f+1))
            # Train network training set
            train_results = self.train(X_train[f], y_train[f], epochs)
            train_fmse.append(train_results['loss']['train'])
            
            # Append weights/bias to new dict
            fold_weights[f+1] = train_results['weights']['split']
            fold_bias[f+1] = train_results['bias']['split']
            
            # Test network with validation set and append MSE + predictions
            val_mse, fold_pred = self.test(X_val[f], y_val[f], self.fnt, self.norm_pred)
            # Metrics
            MetricsCV = Metrics(self.loss_str)
            MetricsCV.load(y_val[f], self.yclass, fold_pred, self.btm)
            cm = MetricsCV.confusion(plot=False)
            
            fold_met.append(cm[1])
            val_fmse.append(val_mse)
            val_true.append(y_val[f])
            val_pred.append(fold_pred)
            
        return {'loss':{'train':train_fmse,'validation':val_fmse},'weights':{'cv':fold_weights},'bias':{'cv':fold_bias},'validation_metrics':fold_met,'cross_val':self.cv_fn, 'data':{'true':val_true, 'prediction':val_pred}}
        
    def test(self, X_test, y_test, efn, normalize):
        """ Get predictions on tested data """
        
        _, y_hat = self.forward_propagate(X_test)
        # Normalize between 0 & 1
        if normalize and (y_hat[self.ll].shape[-1] > 1):
            y_hat[self.ll] = self._normalize(y_hat[self.ll])

        error = efn.fn(y_test, y_hat[self.ll]) 
            
        return error, y_hat[self.ll]
    
    def _normalize(self, data):
        """ Normalize probabilities """
        return data/np.sum(data, axis=1)[:, None]
    
# %% Main
if __name__ == "__main__":    
    from sklearn.model_selection import train_test_split
    from src.crossval import CrossValidationSplit as cvs
    from src.activations import ActivationFunctions as afc
    from src.general import ModelGeneral
    from src.importdata import ImportData
    from src.correlation import CorrelationMethods
    from src.scaler import Scaler
    #%%
    ############################### USER INPUT ###############################
    
    # File (entries-by-features dimensions)
    user_file = "data/sample_data.csv"
    # Entry seperator
    user_file_sep = ','
    # Headers; First line of file; No headers, set to None
    user_file_header = 0
    # Amount of target columns; targets should be last columns (after features)
    user_target_columns = 1
    # Name of the target
    y_feature = "PCa"
    
    # Binary to multiclass target? Converts 1D target to 2D array
    bin_to_multi = True
    
    # Test split ratio or CV
    method = {'split':0.1}# {'split':0.2} ; {'k_fold_strat':{'folds':3}}
    
    # User settings or existing model? Set to None for user settings
    model = 'NN_05-08-2020_20-33-10/model.json' #None
    # User scaler or existing scaler? Set to None for user scaler
    scaler = 'NN_05-08-2020_20-33-10/robust.sav' #None
    
    # Use a scaler?
    user_scaler = "robust" # scaler overrides user_scaler
    
    # Neural network user settings
    user_layers = [5,10,15,20] #[5, 7, 12] FREE PSA  ; [8,13,14] PSA
    user_activations = ['relu','prelu','relu','selu','sigmoid'] #['sigmoid','relu','prelu','sigmoid'] ; ['relu','prelu','isru','sigmoid']
    user_weights_init = ['he_uniform','he_uniform','he_uniform','he_uniform','xavier_uniform'] #['xavier_uniform','he_uniform','he_uniform','xavier_uniform'] ; ['he_uniform','he_uniform','uniform','xavier_uniform'] 
    user_batches = None
    user_cost = Log
    user_optimizer = 'Adam'
    user_optimizer_options = {}
    user_learning_rate = 1e-3
    user_seed=666
    
    # Iterations
    user_iterations = 50000
    
    # Normalize predictions between 0 and 1?
    user_norm_pred = True
    
    # Specify error metric to use for test data
    user_test_metric = Root
    
    # Save test data and predictions? Only applicable for train/test
    save_test_data = True
    
    # Verbose; if False shows only essential model settings
    verbose = True
    
    #################### DO NOT CHANGE BELOW THIS LINE #######################
    #%% 
    # Type train/test or CV
    mkey = list(method.keys())[0]
    if mkey in ['split']:
        method_opt = 'single'
    elif mkey in ['k_fold','k_fold_strat','loo']:
        method_opt = 'multi'
    
    # Data file
    IMD = ImportData(input_file=user_file, encoder_file=False, btm=bin_to_multi, seperator=user_file_sep, header_row=user_file_header, y_col=user_target_columns, verbose=verbose)
    
    # Model General
    MG = ModelGeneral(cwd, IMD.abspath, method, method_opt, verbose)
    
    # Import data + change 1D to 2D multiclass
    X = IMD.X
    y = IMD.y
    if bin_to_multi:
        y_classes = IMD.cats[0].astype(int).tolist()
    else:
        y_classes = np.unique(y).astype(int).tolist()
    
    # Check if model is valid json
    if model is not None:
        user_data = MG.load_model(model)
        user_data['cost'] = globals()[user_data['cost']]
        model_loaded = True
    else:
        # User settings
        user_data = {}
        user_data['activation_functions'] = user_activations
        user_data['batches'] = user_batches
        user_data['cost'] = user_cost
        user_data['gradient_descent'] = user_optimizer
        user_data['gradient_descent_options'] = user_optimizer_options
        user_data['hidden_layers'] = user_layers
        user_data['learning_rate'] = user_learning_rate
        user_data['seed'] = user_seed
        user_data['weights_initialisation'] = user_weights_init
        model_loaded = False
    
    #%% Pearson and pairwise relations
    CORM = CorrelationMethods(IMD.rawdf, IMD.headers, verbose=verbose)
    pearson_result = CORM.pearson(y_feature)
    pairwise_result = CORM.pairwiserelations(y_feature)
    
    #%% Train/Test or CV
    if method_opt == 'single':
        # Train & Test
        
        # Split and shuffle
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=method['split'], random_state=user_seed, shuffle=True)
        # Non scaled copy for Risk Calculator (prostaatwijzer)
        X_test_ns = X_test.copy()
        
        # Check scaler
        SCL = Scaler(cwd, X_train, verbose)
        if scaler is not None:
            # Load external scaler
            SCL.load_scaler(scaler)
            SCL_fn = getattr(SCL, SCL.fn_scaler)
        else:
            SCL_fn = getattr(SCL, user_scaler)
            
        # Scale it
        data_scaler, X_train = SCL_fn()
        X_test = SCL_fn(data_scaler, X_test)
    elif method_opt == 'multi':
        # Cross validation
        
        # Just to make sure
        X_train = X.copy()
        y_train = y.copy()
        
        # Check scaler
        SCL = Scaler(cwd, X_train, verbose)
        if scaler is not None:
            # Load external scaler
            SCL.load_scaler(scaler)
            SCL_fn = getattr(SCL, SCL.fn_scaler)
        else:
            SCL_fn = getattr(SCL, user_scaler)
            
        # Scale it
        data_scaler, X_train = SCL_fn()
    else:
        raise Exception('Invalid method provided. Should contain key `split`, `k_fold`, `k_fold_strat` or `loo`.')

    #%% Initialize network
    NN = NeuralNetwork(X_train, y_train, y_classes, bin_to_multi, user_data['hidden_layers'], user_data['activation_functions'], 
                       user_data['weights_initialisation'], batches=user_data['batches'], cost=user_data['cost'], 
                       optimizer=user_data['gradient_descent'], optimizer_options=user_data['gradient_descent_options'], 
                       learning_rate=user_data['learning_rate'], verbose=verbose, seed=user_data['seed'])
    
    # Set weights of network from loaded model
    if model_loaded:
        NN.weights = user_data['weights']
        NN.bias = user_data['bias']
        
    # %% Train network
    if not model_loaded:
        training_results = NN.training(user_iterations, method, method_opt, IMD.enc, user_test_metric, user_norm_pred)
    
    # %% Save network
    if model_loaded:
        MG.save_model(NN.layers, list(NN.actf.values()), NN.wi, NN.batch_mode, NN.batch_size, 
                      NN.weights, NN.bias, NN.loss, NN.lr, NN.optfn, NN.optvals, NN.seed, bin_to_multi, model)
    else:
        MG.save_model(NN.layers, list(NN.actf.values()), NN.wi, NN.batch_mode, NN.batch_size, 
                      training_results['weights'], training_results['bias'], NN.loss, NN.lr, NN.optfn, NN.optvals, NN.seed, bin_to_multi)
            
    # Also save scaler
    SCL.save_scaler(file_name=MG.current_output_dir)
    
    # And the OneHotEncoder to be sure
    IMD.save_encoder(file_name=MG.current_output_dir)
    # %% Predictions
    
    # Test data
    if method_opt == 'single':
        test_error, pred = NN.test(X_test, y_test, user_test_metric, user_norm_pred)
    
    #%% Metrics
    
    # Save pearson/pairwise plots first    
    MG.save_plot({'pearson':pearson_result['figure'].get_figure(),
                  'pairwise':pairwise_result['figure']})
    
    # For testing
    if method_opt == 'single':
        # Call Metrics and load the test and predictions
        MetricsNN = Metrics(NN.loss_str)
        MetricsNN.load(y_test, y_classes, pred, btm=bin_to_multi)
        
        # Save X_test and predictions as npy
        if save_test_data:
            # Save unscaled test data for RC
            np.save(os.sep.join([MG.current_output_dir,'X_test.npy']), np.concatenate([X_test_ns, np.reshape(np.array(MetricsNN.y_1D),(1, np.array(MetricsNN.y_1D).size)).T], axis=1))
            # Save predictions
            np.save(os.sep.join([MG.current_output_dir,'y_hat.npy']), pred)
        
        # Get binarized test error
        test_error_binarized = user_test_metric.fn(np.array(MetricsNN.y_1D), np.array(MetricsNN.y_hat_1D))
   
        # Error plot can only be done when model was trained, ergo not loaded
        if not model_loaded:
            # Plot training results
            loss_fig = MetricsNN.loss_epoch(training_results)
            # Save the plot
            MG.save_plot({'error_per_epoch':loss_fig})
        
        # Distribution of probability predictions
        hist_fig = MetricsNN.histogram(y_classes)

        # More verbose options for split
        cm, cm_metrics, cm_fig = MetricsNN.confusion(multi=False)
        
        # Sklearns classification report
        report = MetricsNN.class_report()
        if verbose:
            print('\n'+('').center(50,'-')+'\n')
            print('\nClassification report test data:\n')
            print(report['verbose'])
            print('\n'+('').center(50,'-')+'\n')
        
        # ROC / AUC
        AUC, FPR, TPR, roc_fig = MetricsNN.ROC()
        
        # Precision & Recall
        precision, recall, AP_micro, precall_fig = MetricsNN.precision_recall()
        
        if verbose:
            # Accuracy
            print('\n'+(' Metrics ').center(50,'-')+'\n')
            print('Accuracy = {:.2%}'.format(cm_metrics['ACC']))
            print('Balanced accuracy / Non-error rate = {:.2%}'.format(cm_metrics['BACC']))
            print('\n'+('').center(50,'-')+'\n')
        
        # Save confusion metrics
        MG.save_json(data=cm_metrics, fname=os.sep.join([MG.current_output_dir, 'metrics']), load=True)
        
        # Save all plots
        MG.save_plot({'confusion_matrix':cm_fig,
                      'ROC':roc_fig,
                      'precision_recall':precall_fig,
                      'histogram_probability':hist_fig})
    elif method_opt == 'multi':
        if model_loaded:
            raise Exception('Model loading for cross validation models not implemented.')
            
        # Call Metrics
        MetricsNN = Metrics(NN.loss_str)
        
        # Plot training results
        loss_fig = MetricsNN.loss_epoch(training_results)
        cvacc_fig = MetricsNN.cv_accuracy(training_results)
        
        # Save these plots
        MG.save_plot({'error_per_epoch':loss_fig,
                      'cv_fold_accuracy':cvacc_fig})
        
        # Loop over test & predictions per fold/item 
        for idx, pr in enumerate(training_results['data']['prediction']):
            # String for saving purposes concerning fold
            fstr = ['fold', str(idx+1)]
            # Load metrics
            MetricsCVF = Metrics(NN.loss_str)
            MetricsCVF.load(training_results['data']['true'][idx], y_classes, pr, btm=bin_to_multi)
            # Distribution of probability predictions
            hist_fig = MetricsCVF.histogram(y_classes)
            # Do all the confusion, derivations, etc.
            cm, cm_metrics, cm_fig = MetricsCVF.confusion(multi=False)
            AUC, FPR, TPR, roc_fig = MetricsCVF.ROC()
            precision, recall, AP_micro, precall_fig = MetricsCVF.precision_recall()
            # Save confusion metrics
            MG.save_json(data=cm_metrics, fname=os.sep.join([MG.current_output_dir, '-'.join(['metrics']+fstr)]), load=True)
            # Save specific plots
            MG.save_plot({'-'.join(['confusion_matrix']+fstr):cm_fig,
                          '-'.join(['ROC']+fstr):roc_fig,
                          '-'.join(['precision_recall']+fstr):precall_fig,
                          '-'.join(['histogram_probability']+fstr):hist_fig})
            
    print('>> Metrics saved successfully')