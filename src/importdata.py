# -*- coding: utf-8 -*-
"""
Created on Wed May  6 13:19:04 2020

@author: ToshY
"""

import os
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

class ImportData():
    """
    Initial import of user specified dataset

    Attributes
    ----------
    input_file : string
        Filepath of user data
    encoder_file : string or False
        Filepath of optional OneHotEncoder
    btm : bool
        Convert single binary target column to multiclass 2D
    seperator : string
        Delimiter for user data
    verbose : bool
        Console log feedback
    header_row : int
        Remove header row
    y_col : int
        The amount of target columns; last column(s) should be target

    Methods
    -------
    save_encoder(file_name, ext='sav')
        Pickle dump specified encoder
    load_encoder(encoder_file)
        Pickle load specified encoder
    """
    
    def __init__(self, input_file, encoder_file, btm=True, seperator=',', verbose=True, header_row=0, y_col=1):
        """ Import data """
        
        self.abspath = os.path.abspath(input_file)

        # Read into 2D array with header removal
        self.rawdf, self.headers, self.data = self._read_data(input_file, seperator, header_row)
        
        # Verbose?
        self.verbose = verbose
        
        # Amount of y_columns; y_col = amount of initial target columns, not index
        self.y_idx = self.data.shape[1] - y_col
		
        # Seperate X and y
        self.X = self.data[:,:self.y_idx]
        
        # Encoder specified?
        if encoder_file is not False:
            # If not false, load from file name
            self.enc = self.load_encoder(encoder_file)
        else:
            # Default OneHotEncoder will be selected later
            self.enc = False
        
        # Convert 1D binary to multiclass 2D target
        if btm:
            self.y, self.cats = self._binarize( self.data[:,self.y_idx:], self.enc )
            if self.verbose: print('>> Preprocessing target: binary to multiclass')
        else:
            self.y = self.data[:,self.y_idx:]
            if self.verbose: print('>> Preprocessing target: no changes (binary)')
        
        
    def save_encoder(self, file_name, ext='sav'):
        """ Save encoder """
        
        if self.enc is not None and self.enc is not False:
            encoder_filename = os.sep.join([file_name, '.'.join(['OHE', ext])])
            pickle.dump(self.enc, open(encoder_filename, 'wb'))
            if self.verbose:
                print('>> OneHotEncoder saved as `{}`'.format(encoder_filename))
        else:
            print('>> No encoder used; skip saving')
        
    def load_encoder(self, encoder_file):
        """ Load encoder by file """
        
        file = os.sep.join([self.main_model_dir, encoder_file])
        self.enc = pickle.load(open(file, 'rb'))
        
        if self.verbose:
            print('>> External OneHotEncoder loaded for target conversion')
        return self.scaler
        
    def _read_data(self, file, seperator, header_row):
        """ Read data with(out) header row """
        
        df = pd.read_csv(file, sep=seperator, header=header_row)
        return df, list(df), df.to_numpy()
        
    def _binarize(self, output, encoder=None):
        """ Convert binary output to 2D (multiclass) array """
        
        if (output.shape[1] == 1):
            if encoder is False:
                self.enc = OneHotEncoder(handle_unknown='ignore')
            self.enc.fit(output)
            return self.enc.transform(output).toarray(), self.enc.categories_
        
        return output
