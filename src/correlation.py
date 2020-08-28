# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 21:24:47 2020

@author: ToshY
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

class CorrelationMethods():
    """
    Providing correlation metrics Pearson and pairwise relations

    Attributes
    ----------
    user_data : dataframe
        Dataframe of user specified dataset
    user_columns : list
        List of feature columns
    verbose : bool
        Console log feedback

    Methods
    -------
    pearson(target, plot=True, fs=(8,6))
        Pearson correlation with respect to specified target (string)
    pairwiserelations(target, title_extra='initial dataset', colours=['green','red'], marks=['o','D'])
        Pairwise relation plot using Seaborn
    """
    
    def __init__(self, user_data, user_columns, verbose=True):
        # Check if already dataframe
        if not isinstance(user_data, pd.DataFrame):
            self.df = pd.DataFrame(data=user_data,
                              index=None,
                              columns=user_columns)
        else:
            self.df = user_data
        
        # Verbose?
        self.verbose = verbose
        
    def pearson(self, target, plot=True, fs=(8,6)):
        """ Pearson correlation """
        
        cor = self.df.corr()
        #Correlation with output variable
        cor_target = cor[target].sort_values(ascending=False)

        if plot:
            plt.figure(89, figsize=fs)
            h = sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
            h.set_xticklabels(h.get_xticklabels(), rotation=45)
            h.set_yticklabels(h.get_yticklabels(), rotation=45)
            plt.title("Correlation matrix")
        
        if self.verbose: print('>> Pearson correlation plot generated')
        return {'correlation':cor, 'corr_vs_target':cor_target, 'figure':h}

    def pairwiserelations(self, target, title_extra='initial dataset', colours=['green','red'], marks=['o','D']):
        """ Pairwise relationships """
        
        g = sns.pairplot(self.df, height=3, hue=target, palette=sns.xkcd_palette(colours), markers=marks, kind="reg")
        g.fig.suptitle(' '.join(['Pairwise relationships', title_extra]), y=1.03, fontsize=16)
        
        if self.verbose: print('>> Pairwise relation plot generated')    
        return {'figure':g}
