# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 14:47:24 2020

@author: ToshY
"""

import numpy as np
import itertools
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, \
confusion_matrix, precision_recall_curve, roc_curve, auc, \
precision_recall_fscore_support, classification_report, balanced_accuracy_score, \
average_precision_score

class Metrics():
    """
    Providing metrics for a given model by returning the accuracy, precision, 
    recall, F1-score, Cohen's Kappa and a confusion matrix

    Attributes
    ----------
    cost : string
        Name of the cost function (plotting purposes)

    Methods
    -------
    load(y_test, y_classes, y_probabilties, btm=True)
        Load the test and predicted data
    accuracy()
        Returns the accuracy
    precision_recall(plot=True)
        Returns the precision and recall with optional plot
    f1()
        Returns the F1-score
    kappa()
        Returns Cohen's Kappa
    histogram(classes=None, plot=True, fs=(8,6))
        Returns histogram of predicted classes
    confusion(multi=True, plot=True, norm=False, title='Confusion matrix', cmap=plt.cm.coolwarm_r)
        Returns confusion matrix with optional plot
    confusion_derivations(confusion_matrix, multi=True)
        Returns derivations of confusion matrix
    class_report()
        Returns Sklearn's classification report
    cv_accuracy()
        Returns accuracy plots for CV runs
    loss_epoch()
        Returns loss-epoch curve for training or CV data
    ROC(plot=True, fs=(8,6))
        Returns AUC, FPR, TPR and optional curve
    precision_recall(plot=True, fs=(8,6))
        Returns precision, recall, average_precision["micro"] and optional curve
        
    """
    
    def __init__(self, cost):
        # Loss string
        self.loss_method = cost
        
    
    def load(self, y_test, y_classes, y_probabilties, btm=True):
        # Target y-values; sample-by-class 
        self.y = y_test
        
        # Amount of classes; for plotting purposes
        self.yc = y_classes
        self.len_classes = self.y.shape[1]
        
        # Predicted y-values; sample-by-class        
        self.y_prob = y_probabilties
        
        # Classify
        if btm:
            # Initially multiclass
            self.y_1D, self.y_hat_1D, self.y_hat_1D_raw = self._classify()
        else:
            # Initially binary
            self.y_1D, self.y_hat_1D, self.y_hat_1D_raw = self._classify_1D()

    def accuracy(self):
        return accuracy_score( self.y_1D, self.y_hat_1D )
    
    def balanced_accuracy(self):
        return balanced_accuracy_score( self.y_1D, self.y_hat_1D )
       
    def f1(self):
        return f1_score( self.y_1D, self.y_hat_1D )
    
    def kappa(self):
        return cohen_kappa_score( self.y_1D, self.y_hat_1D )
    
    def histogram(self, classes=None, plot=True, fs=(8,6)):
        """ Histogram plot """
        
        # Plot colours
        col = cycle('grbcmk')
        kwargs = dict(alpha=0.5, bins=100)
        
        plt.figure(94, figsize=fs)
        fig = plt.gcf()
        for i in range(self.y_prob.shape[-1]):               
            plt.hist(self.y_prob[:,i], **kwargs, color=next(col), label=classes[i])
        
        
        plt.title("Distribution of predicted probabilities")
        plt.ylabel('Frequency')
        plt.xlabel('Probability')
        plt.xlim(0,1)
        plt.xticks(self._listrange(0,1,1/10))
        if self.y_prob.shape[-1] > 1:
            plt.legend(loc="best")
        plt.show()
        
        return fig
        
    def confusion(self, multi=True, plot=True, norm=False, title='Confusion matrix', cmap=plt.cm.coolwarm_r):
        """ Return confusion matrix with plot option"""
        
        # The confusion matrix
        cm = confusion_matrix(self.y_1D, self.y_hat_1D)
        # Derivations
        cm_metrics = self.confusion_derivations(cm, multi)
        
        # Normalize
        if norm:
            cm = self._normalize(cm)

        # Check for plot options
        if plot:
            if self.yc is not None:
                classes = self.yc
            else:
                classes = sorted([*{*self.y_1D}])
            
            plt.figure(97)
            fig = plt.gcf()
            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            tick_marks = np.array(self.yc)
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)
        
            fmt = '.2f' if norm else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")
        
            plt.ylabel('Target')
            plt.xlabel('Prediction')
            plt.show()
        else:
            fig = None
    
        return cm, cm_metrics, fig
    
    def confusion_derivations(self, confusion_matrix, multi=True):
        """ Get derivations of confusion matrix """

        # Basic derivations
        if confusion_matrix.shape == (2,2) and multi is False:
            # Binary
            TN, FP, FN, TP = confusion_matrix.ravel()
        else:
            # Multiclass
            FP = (confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)).astype(float)
            FN = (confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)).astype(float)
            TP = (np.diag(confusion_matrix)).astype(float)
            TN = (confusion_matrix.sum() - (FP + FN + TP)).astype(float)
        
        P = (TP+FN).astype(float)
        N = (TN+FP).astype(float)
        
        # Add everything to dictonary
        metrics = {'P':P.astype(int),'N':N.astype(int), \
                   'TP':TP.astype(int),'FP':FP.astype(int),\
                   'TN':TN.astype(int),'FN':FN.astype(int)}
        # Recall
        metrics['TPR'] = TP/P
        # Specificicty
        metrics['TNR'] = TN/N
        # Precision
        metrics['PPV'] = TP/(TP+FP)
        # Negative predictive value
        metrics['NPV'] = TN/(TN+FN)
        # False negative rate
        metrics['FNR'] = 1-metrics['TPR']
        # False positive rate
        metrics['FPR'] = 1-metrics['TNR']
        # False discovery rate
        metrics['FPR'] = 1-metrics['PPV']
        # False Omission rate
        metrics['FOR'] = 1-metrics['NPV']
        # Critical Success Index
        metrics['TS'] = TP/(TP+FN+FP)
        # Accuracy
        metrics['ACC'] = (TP+TN)/(P+N)
        # Balanced Accuracy
        metrics['BACC'] = (metrics['TPR']+metrics['TNR'])/2
        # Predicted positive condition rate
        metrics['PPCR'] = (TP+FP)/(TP+FP+TN+FN)
        # F1-score
        metrics['F1'] = 2*(metrics['PPV']*metrics['TPR'])/(metrics['PPV']+metrics['TPR'])
        # Matthews correlation coefficient
        metrics['MCC'] = ((TP*TN)-(FP*FN))/(np.sqrt(((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))))
        # Fowlkes-Mallows index
        metrics['FM'] = np.sqrt(metrics['PPV']*metrics['TPR'])
        
        # Return metrics
        return metrics
    
    def class_report(self):
        """ Overview of precision, recall, accuracy and f1-score """
        
        PPV, TPR, F1, support = precision_recall_fscore_support(self.y_1D, self.y_hat_1D)
        
        # Console printable version
        report = classification_report(self.y_1D, self.y_hat_1D)
        return {'PPV':PPV,'TPR':TPR,'F1':F1,'Support':support,'verbose':report}
    
    def cv_accuracy(self, train_results, plot=True, fs=(8,6)):
        """ Accuracy for cross validation sets """

        if plot:
            plt.figure(95, figsize=fs)
            fig = plt.gcf()
            
            plist = []
            for m, v in enumerate(train_results['validation_metrics']):
                plist.append(v['ACC'])
            
            plt.plot(list(range(1,m+2)),plist,lw=2)
            plt.title("Accuracy {} cross validation ({}-fold)".format(train_results['cross_val'].replace('_','-').title(), m+1))
            plt.xlabel("Fold")
            plt.xticks(list(range(1,m+2)))
            plt.ylabel('Accuracy')
            plt.show()
        else:
            fig = None
        
        return fig
        
    def loss_epoch(self, train_results, plot=True, fs=(8,6)):
        """ Plot error per epoch """
        
        error = train_results['loss']
        # Check if CV
        if len(error)>1:
            cv = True
        else:
            cv = False
        
        if plot:
            plt.figure(96, figsize=fs)
            fig = plt.gcf()
            
            if cv:
                 # CV training
                 for p in range(len(error['train'])):
                     plt.plot(*zip(*error['train'][p]),lw=2,label='fold {}'.format(p+1))
                 
                 plt.title("{} per epoch ({}-fold CV)".format(self.loss_method, len(error['validation'])))
                 
                 # Only show legend if it fits; +/- <10
                 if len(error['train']) <= 10:
                     plt.legend(loc="best")
                     
            else:
                 # Normal training
                 for p in range(1):
                     plt.plot(*zip(*error['train']),lw=2,label='error')
                     
                 plt.title("{} per epoch".format(self.loss_method, len(error)))
             
            plt.xlabel("Epoch")
            plt.ylabel(self.loss_method)
            plt.show()
        else:
            fig = None
                
        return fig
    
    
    def ROC(self, plot=True, fs=(8,6)):
        """ TPR and FPR with optional plot """
        
        # Plot colours
        col = cycle('grbcmk')
        
        fpr = {}
        tpr = {}
        roc_auc = {}
        
        if plot:
            plt.figure(98, figsize=fs)
            fig = plt.gcf()
        else:
            fig = None
        
        for i in range(self.len_classes):
            fpr[i], tpr[i], _ = roc_curve(self.y[:, i], self.y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
            if plot and self.len_classes == 1:
                clabel = 'ROC curve'
            else:
                clabel = 'class {}'.format(i)
            plt.plot(fpr[i], tpr[i], lw=2, label=clabel+' (AUC = {})'.format(round(roc_auc[i],3)), color=next(col))
                   
        # Micro average
        roc_auc['micro'], fpr['micro'], tpr['micro'] = self._micro_roc(fpr, tpr)
        
        # Macro average
        roc_auc['macro'], fpr['macro'], tpr['macro'] = self._macro_roc(fpr, tpr)
        
        if plot:
            if self.len_classes > 1:
                # Micro average 
                plt.plot(fpr['micro'], tpr['micro'], label='micro-average (AUC = {0:0.3f})' ''.format(roc_auc['micro']), color='deeppink', linestyle=(0, (1, 1)), lw=3)
            # Macro average
            plt.plot(fpr['macro'], tpr['macro'], label='macro-average (AUC = {0:0.3f})' ''.format(roc_auc['macro']), color='navy', linestyle=(0, (1, 1)), lw=2)
            # Add no skill line
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='No skill (AUC = 0.500)')
            plt.xlim([0.0, 1.0])
            plt.xlabel("FPR")
            plt.ylim([0.0, 1.05])
            plt.ylabel("TPR")
            plt.legend(loc="best")
            plt.title("ROC curve")
            plt.show()
    
        return roc_auc, fpr, tpr, fig

    def _micro_roc(self, FPR={}, TPR={}):
        """ Compute micro-average ROC curve and AUC """
        
        FPR, TPR, _ = roc_curve(self.y.ravel(), self.y_prob.ravel())
        
        return auc(FPR, TPR), FPR, TPR
    
    def _macro_roc(self, FPR, TPR):
        """ Compute macro-average ROC curve and AUC """
        
        # First aggregate all false positive rates
        FPR_all = np.unique(np.concatenate([FPR[i] for i in range(self.len_classes)]))
        
        # Interpolate all ROC curves at this points
        TPR_mean = np.zeros_like(FPR_all)
        for i in range(self.len_classes):
            TPR_mean += np.interp(FPR_all, FPR[i], TPR[i])
        
        # AUC by averaging
        TPR_mean /= self.len_classes

        return auc(FPR_all, TPR_mean), FPR_all, TPR_mean 
    
    def precision_recall(self, plot=True, fs=(8,6)):
        """ Precision and recall for given predicted classes """
        
        # Plot colours
        col = cycle('grbcmk')

        average_precision = {}
        precision = {}
        recall = {}
        
        if plot:
            plt.figure(99, figsize=fs)
            fig = plt.gcf()
        else:
            fig = None
        
        for i in range(self.len_classes):
            precision[i], recall[i], _ = precision_recall_curve(self.y[:, i],self.y_prob[:, i])
            average_precision[i] = average_precision_score(self.y[:, i],self.y_prob[:, i])
            plt.plot(recall[i], precision[i], lw=2, label='class {}'.format(i), color=next(col))
        
        # Micro
        precision["micro"], recall["micro"], _ = precision_recall_curve(self.y.ravel(),self.y_prob.ravel())
        average_precision["micro"] = average_precision_score(self.y, self.y_prob, average="micro")
        
        if plot:
            if self.len_classes > 1:
                plt.plot(recall['micro'], precision['micro'], label='micro average', color='deeppink', linestyle=(0, (1, 1)), lw=2)
            plt.xlim([0.0, 1.0])
            plt.xlabel("Recall")
            plt.ylim([0.0, 1.05])
            plt.ylabel("Precision")
            if self.len_classes > 1:
                plt.legend(loc="best")
            plt.title("Precision vs. Recall curve (AP={0:0.2f})".format(average_precision["micro"]))
            plt.show()
            
        return precision, recall, average_precision["micro"], fig
        
    def _classify(self):
        """ Returns max probability by largest argument """
        
        y_true = []
        y_pred_raw = []
        y_pred = []
        for i in range(len(self.y)):
            y_true.append(np.argmax(self.y[i]))
            y_pred_raw.append(max(self.y_prob[i]))
            y_pred.append(np.argmax(self.y_prob[i]))
            
        return y_true, y_pred, y_pred_raw
    
    def _classify_1D(self):
        """ Returns 0 if X < 0.5; Returns 1 if X >= 0.5 """
        
        yprobin = self.y_prob.copy()
        
        yprobin[yprobin<0.5] = 0
        yprobin[yprobin>=0.5] = 1
        
        return self.y.flatten().tolist(), yprobin.flatten().tolist(), self.y_prob.flatten().tolist()
        
    def _normalize(self, cm):
        return cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    def _listrange(self, start=0, end=1, step=1/10):
        return [round(num, 2) for num in np.linspace(start,end,(end-start)*int(1/step)+1).tolist()]
