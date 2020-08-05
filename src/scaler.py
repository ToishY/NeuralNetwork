# -*- coding: utf-8 -*-
"""
Created on Fri May  1 14:40:31 2020

@author: ToishY
"""

import os
import inspect
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler,\
RobustScaler

class Scaler():
    """
    Providing several scalers from Sklearn to use as preprocessing step
    
    
    Attributes
    ----------
    main_dir : array
        Current working directory
    X_training_data : array
        The X training data
    verbose : bool
        Console log feedback

    Methods
    -------
    save_scaler(file_name, ext='sav')
        Save the user specified scaler
    load_scaler(scaler_file)
        Load the user specified scaler
    standard(cscaler=None, X_test=None)
        Returns standard scaled data
    minmax(cscaler=None, X_test=None)
        Returns min/max scaled data
    maxabs(cscaler=None, X_test=None)
        Returns max absolute scaled data
    robust(cscaler=None, X_test=None)
        Returns robust scaled data
    """
    
    def __init__(self, main_dir, X_training_data, verbose=True):
        # Current directory
        self.cwd = main_dir
        # Models directory
        self.main_model_dir = os.sep.join([self.cwd, 'models'])
        # X
        self.X_train = X_training_data
        self.scaler = None
        # Verbose?
        self.verbose = verbose
        
    def save_scaler(self, file_name, ext='sav'):
        """ Save scaler """
        
        if self.scaler is not None:
            scaler_filename = os.sep.join([file_name, '.'.join([self.fn_scaler, ext])])
            pickle.dump(self.scaler, open(scaler_filename, 'wb'))
            if self.verbose:
                print('>> Scaler saved as `{}`'.format(scaler_filename))
        else:
            raise Exception('Cannot save NoneType. Please specify a scaler first.')
        
    def load_scaler(self, scaler_file):
        """ Load scaler by file """
        
        file = os.sep.join([self.main_model_dir, scaler_file])
        
        self.scaler = pickle.load(open(file, 'rb'))
        self.fn_scaler = os.path.splitext(os.path.basename(file))[0]
        
        print('>> External data scaler loaded for data manipulation')
        return self.scaler
            
    def standard(self, cscaler=None, X_test=None):
        """ Standard scaling
        
        Standardize features by removing the mean and scaling to unit variance
        
        """
        
        if cscaler is None or X_test is None:
            self.scaler = StandardScaler()
            self.fn_scaler = inspect.stack()[0][3]
            return self.scaler, self.scaler.fit_transform(self.X_train)
        else:
            return cscaler.transform(X_test)
          
    def minmax(self, cscaler=None, X_test=None):
        """ MinMax scaling
        
        Transform features by scaling each feature to a given range, defaults
        to [0, 1]
        
        """
        
        if cscaler is None or X_test is None:
            self.scaler = MinMaxScaler()
            self.fn_scaler = inspect.stack()[0][3]
            return self.scaler, self.scaler.fit_transform(self.X_train)
        else:
            return cscaler.transform(X_test)
    
    def maxabs(self, cscaler=None, X_test=None):
        """ MaxAbs scaling
        
        Scale each feature by its maximum absolute value so in a way that 
        the training data lies within the range [-1, 1]
        
        """
        if cscaler is None or X_test is None:
            self.scaler = MaxAbsScaler()
            self.fn_scaler = inspect.stack()[0][3]
            return self.scaler, self.scaler.fit_transform(self.X_train)
        else:
            return cscaler.transform(X_test)
    
    def robust(self, cscaler=None, X_test=None):
        """ Robust scaling
        
        Scale features using statistics that are robust to outliers.
        This Scaler removes the median and scales the data according to the 
        quantile range, defaults to IQR
        
        """
        
        if cscaler is None or X_test is None:
            self.scaler = RobustScaler()
            self.fn_scaler = inspect.stack()[0][3]
            return self.scaler, self.scaler.fit_transform(self.X_train)
        else:
            return cscaler.transform(X_test)