# -*- coding: utf-8 -*-
"""
Created on Sun Jun  7 15:54:43 2020

@author: ToishY
"""

import os
import json
import datetime
import numpy as np
import pandas as pd

class ModelGeneral():
    """ 
    Saving and loading of (existing) models
    
    Attributes
    ----------
    main_dir : string
        Current working directory
    file_path : string
        Filepath of user data
    method : dict
        Dictonary specifying train/test split or k-fold/loo CV
    method_opt : string
        Specifying 'single' for train/test or 'multi' for k-fold/loo CV
    verbose : bool
        Console log feedback
    prefix : string
        Optional prefix for naming of output folders in output directory

    Methods
    -------
    save_model(layers, ..., model=False)
        Save the model used at runtime; automatic
    save_readme(input_file, method, seed, btm=False, model=False)
        Add additional information about model used at runtime in a TXT file
    load_model(model_file)
        Use an existing model for current run
    save_json(data, fname, load=False)
        Save data as (beautified) JSON
    save_plot(self, figures, ext='png', dpires=300, transp=True)
        Save high-resolution plots from specified dict 'figures'
    """
    
    def __init__(self, main_dir, file_path, method, method_opt, verbose=True, prefix='NN_'):
        # Input file
        self.input = file_path
        
        # Idenitify method; split or CV
        self.method = method
        self.mopt = method_opt
            
        # Set datetime for models/plots
        self.run_date = datetime.datetime.now()
        self.run_date_format = self.run_date.strftime("%d-%m-%Y_%H-%M-%S")
        self.prefix = prefix

        # Current directory
        self.cwd = main_dir
		
        # Output directoreis
        self.main_data_dir = os.sep.join([self.cwd, 'data'])
        self.main_model_dir = os.sep.join([self.cwd, 'models'])
        self.main_plot_dir = os.sep.join([self.cwd, 'output'])      
        
        # Verbose?
        self.verbose=verbose
            
    def save_model(self, layers, activation_functions, weights_inits, batch_mode, batches, weights, bias, cost, learning_rate, optimizer, optimizer_options, seed, btm=False, model=False):
        """ Prepare data for saving """
        
        # Output directory for this run + create subdirectories
        self.current_output_dir = os.sep.join([self.main_model_dir, (self.prefix+self.run_date_format)])
        self._create_directories( [self.current_output_dir] )         
        
        # File
        file_name = os.sep.join([self.current_output_dir, 'model'])
        
        # Data
        data = {}
        data['hidden_layers'] = layers[1:-1]
        data['activation_functions'] = activation_functions
        data['weights_initialisation'] = weights_inits
        if batch_mode:
            data['batches'] = batches
        else:
            data['batches'] = str(batch_mode).lower()
        data['cost'] = str(cost.__name__)
        data['iterations'] = (optimizer_options['epoch']-1)
        data['gradient_descent'] = str(optimizer.__name__)
        data['gradient_descent_options'] = {k: optimizer_options[k] for k in set(list(optimizer_options.keys())) - set(['epoch'])}
        data['learning_rate'] = learning_rate
        
        # method
        mkey = list(weights.keys())[0]
        if mkey in ['split']:
            # Split
            data['weights'] = json.loads(self._jsonify(weights['split'], 'values'))
            data['bias'] = json.loads(self._jsonify(bias['split'], 'values'))
        elif mkey in ['cv']:
            # CV
            fold_weight = {}
            fold_bias = {}
            for (k1, w), (k2, b) in zip(weights['cv'].items(), bias['cv'].items()):
                fold_weight['fold-'+str(k1)] = json.loads(self._jsonify(w, 'values'))
                fold_bias['fold-'+str(k2)] = json.loads(self._jsonify(b, 'values'))
            
            data['weights'] = fold_weight
            data['bias'] = fold_bias
            
        data['seed'] = seed

        # Dump beautified json
        jdata = json.loads(json.dumps(data))
        self.save_json(data=jdata, fname=file_name)
        
        # Save README
        self.save_readme(self.input, self.method, seed, btm, model)
        
        if self.verbose:
            print('\n>> Model saved in directory `{}`'.format(self.current_output_dir ))
    
    def save_readme(self, input_file, method, seed, btm=False, model=False):
        """ Create README file with input file """
        
        # Method used
        if self.mopt == 'single':
            method_str = 'split = ' + str(method['split'])
        elif self.mopt == 'multi':
            key = list(method.keys())[0]
            val = list(method[key].values())[0]
            method_str = key.replace('_','-').title() + ' = ' + str(val) + ' folds'
            
        # Was it a previous model? Add information
        if type(model) is not bool:
            prev_model = 'Previous model: ' + os.path.abspath(model)
        else:
            prev_model = ''
        
        # Create list (+remove empty)
        strlist = list(filter(None,
                              ['Input file: '+input_file, 
                               prev_model,
                               'Method: '+method_str, 
                               'Target binary to multiclass conversion: ' + str(btm),
                               'Seed: '+str(seed)]))
        
        # README
        readme_file =  os.sep.join([self.current_output_dir, 'README.txt'])
        with open(readme_file, 'w') as fh:
            fh.write("\n".join(str(item) for item in strlist))
    
    def load_model(self, model_file):
        """ Load a model from JSON """
        
        model_path = os.sep.join([self.main_model_dir, model_file])
        self.prev_model = os.path.abspath(model_path)
        
        with open(model_path) as json_file:
            data = json.load(json_file)
            
        
        template = ['activation_functions','batches','bias','cost',
                    'gradient_descent','gradient_descent_options','hidden_layers',
                    'iterations','learning_rate','seed','weights','weights_initialisation']
            
        if all(fn in list(data.keys()) for fn in template):
            if data['batches'] == 'false':
                data['batches'] = None
                
            # Load weights and biases into acceptable model format
            wdat = {}
            bdat = {}
            for l, itm in enumerate(data['weights']):
                wdat[l+1] = np.array(itm)
                bdat[l+1] = np.array(data['bias'][l])
            
            data['weights'] = wdat
            data['bias'] = bdat
            
            if self.verbose:
                print('>> External model loaded for network setup')
                
            return data
        else:
            raise Exception('The provided JSON file did not contain all valid keys needed for model initialisation.')
    
    def save_json(self, data, fname, load=False):
        """ Dump beautiful json """
        
        # Load first?
        if load:
            data = json.loads(json.dumps(data, default=float))
        
        ufile = fname+'.json'
        f = open(ufile, "w")
        json.dump(data, f, ensure_ascii=False, sort_keys=True, indent=4)
        f.close()
    
    def save_plot(self, figures, ext='png', dpires=300, transp=True):
        """ Save plots in dictonary {file_name:fig} """
        
        self.current_plot_dir = os.sep.join([self.current_output_dir, 'plots'])
        self._create_directories([self.current_plot_dir])    
        
        # Iterate over dict and save figs
        for k in figures:
            ufig = figures[k]
            if ufig is not None:
                fname = '.'.join([os.sep.join([self.current_plot_dir, k]), ext])
                ufig.savefig(fname, dpi=dpires, transparent=transp, bbox_inches='tight', pad_inches=.25)
        
    def _jsonify(self, data, orientation='split'):
        """ JSONIFY data """

        if not (data is None):
            return pd.Series(data).to_json(orient=orientation)
        else:
            # DeprecationWarning for None
            return pd.Series(data, dtype=object).to_json(orient=orientation)
    
    def _create_directories(self, dirs, access='755'):
        """ Create new directories """
        
        for d in dirs:
            if not os.path.exists(d) :
                os.mkdir(d, int(access.zfill(4)))