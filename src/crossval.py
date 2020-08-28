# -*- coding: utf-8 -*-
"""
Created on Tue Apr 21 22:33:43 2020

@author: ToshY
"""

from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut

class CrossValidationSplit():
    """
    Cross validation methods for splitting data into train and test sets

    Attributes
    ----------
    X_values : array
        The X-values / features of a given dataset
    predicted_result : array
        The Y-values / outcome of a given dataset

    Methods
    -------
    k_fold(splits=10, rndm_state=None, shffl=True)
        Returns amount of splits & train/test data for K-Fold CV
    k_fold_strat(splits=10, rndm_state=None, shffl=True)
        Returns amount of splits & train/test data for Stratified K-Fold CV
    loo(splits=10)
        Returns amount of splits & train/test data for Leave One Out CV
    """
    
    def __init__(self, X_values, Y_values):
        self.X = X_values
        self.y = Y_values
    
    def k_fold(self, opts={'folds':10,'seed':None,'shuffle':True}):
        kfcv = KFold( n_splits=opts['folds'], random_state=opts['seed'], shuffle=opts['shuffle'] )
        return kfcv.get_n_splits(self.X), self._iterate( kfcv.split( self.X ) )
    
    def k_fold_strat(self, opts={'folds':10,'seed':None,'shuffle':True}):
        kfscv = StratifiedKFold(n_splits=opts['folds'], random_state=opts['seed'], shuffle=opts['shuffle'] )
        return kfscv.get_n_splits(self.X), self._iterate( kfscv.split( self.X, self.y ) )
    
    def loo(self, opts={}):
        loocv = LeaveOneOut()
        return loocv.get_n_splits(self.X), self._iterate( loocv.split( self.X ) )
    
    def _iterate(self, cv):
        """ Add splits to 2D array """
        
        X_train = []
        X_val = []
        y_train = []
        y_val = []
        for train_index, test_index in cv:
            X_train.append(self.X[train_index])
            y_train.append(self.y[train_index])
            X_val.append(self.X[test_index])
            y_val.append(self.y[test_index])
        return X_train, y_train, X_val, y_val
