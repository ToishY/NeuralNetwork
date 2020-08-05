# NeuralNetwork
Neural network for binary and multiclass classification

## Description

Setup a neural network for binary and multiclass classification "from scratch" in Spyder with Python 3.7.6.

This model was created by adapting and combining a wide variety of snippets in order to create an easy to setup multi layer perceptron. Credits to information sources are available in `main.py`. A simple example on how backpropagation works can also be found there.

## Data

Supplied artificial sample data (CSV) for demonstration purposes.

**Features**
* Age
* PSA (µg/L)
* Prostate volume (mL)
* PIRADS score (1-5)

**Target**
* PCa (0/1)

The [Scikit-Learn datasets](https://scikit-learn.org/stable/datasets/index.html) `make_moons` and `load_digits` also have been tested.

## Setup

The scripts were made in Spyder with Python 3.7.6 and the following packages installed

* Numpy 1.18.1
* Scikit-Learn 0.22.1
* MatPlotLib 3.1.3

## User inputs

The following inputs are required from the user:

* `user_file` **[str]** : the complete data file for training and testing the network.
* `user_file_sep` **[str]** : the delimiter for the corresponding data.
* `user_file_header` **[int]** : the line numbber of the header if present; if not present supply `None`.
* `user_target_columns` **[int]** : Amount of target columns. The target columns should be at the end of the feature columns.
* `y_feature` **[str]** : The header name for the y feature (plotting purposes).
* `bin_to_multi` **[bool]** : Specify if binary target column should be converted to multiclass; If `False`, 1 output neuron will be present; If `True`, 2 output neurons will be present.
* `method` **[dict]** : Specify to do Train/Test split or to use Cross-Validation (CV); If train/test, supply `{'split':0.2}`; If CV, supply `{'k_fold':{'folds':3}}`, `{'k_fold_strat':{'folds':3}}` or `{'loo':{}}`.
* `model` **[str]** : Specify if user wants to use an existing model; Specify the `JSON` file to be used or supply `None` if the user wants to use current model settings. E.g. `'NN_05-08-2020_20-33-10/model.json'`.
* `scaler` **[str]** : Specify if user wants to use an existing scaler; Specify the `SAV` file to be used or supply `None` if the user wants to use current scaler settings. E.g. `'NN_05-08-2020_20-33-10/robust.sav'`.
* `user_scaler` **[str]** : Specify the scaler the user wants to use; Valid options are: `"standard"`, `"minmax"`, `"maxabs"`, `"robust"`.
* `user_layers` **[list]** : Specify a list with the amount of neurons per layer / list entry. E.g. : `[8,13,14]`.
* `user_activations` **[list]** : Specify a list with activation functions to use in between layers. E.g. : `['relu','prelu','isru','sigmoid']`.
* `user_weights_init` **[list]** : Specify a list of weight initialisations. E.g. : `['he_uniform','he_uniform','uniform','xavier_uniform']`.
* `user_batches` **[int]** : Specify if the user wants to use batches. If `None`, the entire batch will be used.
* `user_cost` **[obj]** : Specify the cost/loss function to use. Valid options are: `Log` or `Quadratic`.
* `user_optimizer` **[str]** : Specify the gradient descent optimizer function. Valid options are: `"GD"`, `"Momentum"`, `"Adam"`.
* `user_optimizer_options` **[dict]** : Specify options for gradient descent optimizers. Available for `Momentum` and `Adam`. E.g. `{'gamma':0.9}` and `{beta_1':0.9, 'beta_2':0.999, 'eps':1e-8}` resp.
* `user_learning_rate` **[float]** : The learning rate.
* `user_seed` **[int]** : The initial seed.
* `user_iterations` **[int]** : The amount of iterations to train the network.
* `user_norm_pred` **[bool]** : Normalize predictions between 0 and 1 (for multiclass classification).
* `user_test_metric` **[obj]** : Error metric for test data. Valid options are: `Log`, `Quadratic`, `Root.
* `save_test_data` **[bool]** : Save the test data and prediction for later usage? Specify `True` or `False`.
* `verbose` **[bool]** : Verbose mode? If `True`, all possible statements will be logged to console; If `False`, only essential information will be logged to console.

## Info 

### Scaling

In general it's a good idea to scale the data before training the network. Here scaling is performed by using Scikit-learn's toolboxes. Because there is no "optimal" scaler to be used, it's up to the user to decide which scaling method fits best for the specific data. For more information, please check the the different [`scaler methods`](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_all_scaling.html).

### Train/Test

In general it's a good idea to split the data into a train set and test set, with a ratio of 80:20. The splitting is performed by using Scikit-learn's [`train_test_split`](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). In `main.py` this is denoted by the following:
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=user_seed, shuffle=True)
```

***Note***: `shuffle=True` can be manually changed and is not specified as one of the earlier defined `user_` settings.

### Cross Validation

In general it's a good idea to perform cross-validation to validate model stability. It is opted to use `Leave-One-Out` or `Stratified K-Fold` when the sample size is small. The advantage of *Stratified K-Fold* over *Leave-One-Out* is that the population of each fold will better reflect the original population. Please note that when the sample size is small and one target feature is significantly more present than the other, that too many folds can lead to a lower accuracy in some of these folds. For more information, please check the the different [`cross validation methods`](https://scikit-learn.org/stable/modules/cross_validation.html).

## Intialize the network

The network is initialized with the earlier specified user settings like so:

```python
    NN = NeuralNetwork(X_train, 
                       y_train,
                       y_classes,
                       bin_to_multi, 
                       user_data['hidden_layers'], 
                       user_data['activation_functions'], 
                       user_data['weights_initialisation'], 
                       batches=user_data['batches'], 
                       cost=user_data['cost'], 
                       optimizer=user_data['gradient_descent'], 
                       optimizer_options=user_data['gradient_descent_options'], 
                       learning_rate=user_data['learning_rate'], 
                       verbose=verbose, 
                       seed=user_data['seed'])

```

## Training

The network will be trained with the user specified settings

```python
training_results = NN.training(user_iterations, method, method_opt, IMD.enc, user_test_metric, user_norm_pred)
```

Returns the `training_results`, a dictionary containing the training error, weights and biases.

For **train/test** runs, *training_results* will return the following format: `{'loss':{'train':mse},'weights':{'split':self.weights},'bias':{'split':self.bias}}`

For **CV** runs, *training_results* will return the following format: `{'loss':{'train':train_fmse,'validation':val_fmse},'weights':{'cv':fold_weights},'bias':{'cv':fold_bias},'validation_metrics':fold_met,'cross_val':self.cv_fn, 'data':{'true':val_true, 'prediction':val_pred}}`

***Note***: `IMD.enc` denotes a possible [(OneHot) encoder](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html).

## Predictions

After training the network, predictions can be made on the test data (**train/test** runs only).

```python
test_error, pred = NN.test(X_test, y_test, user_test_metric, user_norm_pred)
```

This returns the error and predictions for the test data. 

## Output

All output metrics are saved to a subfolder in the folder `models`, specified by a datetime, e.g. `NN_05-08-2020_20-33-10`.

* `models/NN_05-08-2020_20-33-10`
   * `plots`
      * `confusion_matrix.png`
      * `error_per_epoch.png`
      * `histogram_probability.png`
      * `pairwise.png`
      * `pearson.png`
      * `precision_recall.png`
      * `ROC.png`
   * `metrics.json` : Derivations of the confusion matrix
   * `model.json` : Model parameters
   * `OHE.sav` : OneHotEncoder
   * `README.txt` : Information of input data, split/cv, binary-multiclass conversion and seed
   * `robust.sav` : Scaler method
   * `X_test.npy` : Test data
   * `y_hat.npy` : Test predictions


***Note***: `pearson.png` and `pairwise.png` represent Pearson correlation and pairwise relationship plots. For more information, please check the [pairplot method](https://seaborn.pydata.org/generated/seaborn.pairplot.html).

---

## TO DO

There's still some things which could be improved:
* Automatically determine the appropriate amount of layers/neurons (by using K-Fold CV)
* Add more gradient descent optimizers (e.g. RMSProp, Adagrad, etc.)
* A nice touch would be an easy to use (web) interface

### Afterword

This model was made to gain more knowledge about neural networks and its inner workings. 
