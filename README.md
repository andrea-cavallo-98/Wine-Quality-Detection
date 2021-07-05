# Wine-Quality-Detection

## Project organization

The project is organized in four folders:

* `Data`: contains training and test data, plus many other data there were stored during the realization of the different plots and functions (in the form of numpy arrays)
* `Images`: contains the images generated by the code, which are also reported in the report
* `Results`: contains the results of different parts of the code, stored in textual files. The most relevant of these results are present in the report
* `src`: contains the code 

Then, the report for the project is available in the main folder.



## Code organization

The code is divided into different files.

* `load_data.py`: contains the functions to load the dataset and to split it when needed (two splits, one containing 80% of the data, the other 20%), plus some information on the data (such as attribute names, number of classes...)

* `data_visualization.py`: contains functions to visualize the features and perform some preprocessing. In particular, it allows to:
  * print histograms of the features
  * gaussianize features, both for the training and for the test set
  * calculate the correlation among features and print it with a heatmap
  * perform Z-normalization, both for the training and for the test set
* `pca.py`: contains the functions to compute PCA over a dataset, both for training and for test data (which require to project the test data on the dimensions evaluated from the training data).
* `gaussian_models.py`: contains the functions to train different Gaussian models (MVG, naive Bayes, tied MVG and tied naive Bayes) and to evaluate their performances through k-fold cross validation. The main function trains the 4 Gaussian models on differently pre-processed data (raw, Z-normalized, gaussianized, with PCA or not) and calculates the min DCF using both 5-fold cross validation and a single split approach. Results are stored in a textual file.
* `logistic_regression.py`: contains the functions to train logistic regression models (linear and quadratic) and to evaluate their performances through k-fold cross validation. The main function trains some models on differently pre-processed data (raw, Z-normalized, gaussianized, with PCA or not) using different values of the hyperparameters and calculates the min DCF using both 5-fold cross validation and a single split approach. Results are stored in a textual file.
* `support_vector_machines.py`: contains the functions to train SVM models (linear, quadratic and RBF kernel) and to evaluate their performances through k-fold cross validation. The main function trains the 3 types of SVM using different values of the hyperparameters (and also with and without class rebalancing, on raw, Gaussianized and Z-normalized features) and calculates the min DCF using both 5-fold cross validation and a single split approach. Results are stored in a textual file.
* `gmm.py`: contains the functions to train GMM classifiers (standard, tied or diagonal) and to evaluate their performances through k-fold cross validation. The main function trains different GMM classifiers on Z-normalized and Gaussianized features using several numbers of components and calculates the min DCF using 5-fold cross validation. Results are stored in a textual file.
* `prediction_measurement.py`: contains functions to analyse model performances and do some plots. In particular, it allows to:
  * calculate from the log-likelihood ratios the minimum DCF and the actual DCF using the theoretical threshold
  * print ROC curves and Bayes error plots from the log-likelihood ratios
  * calculate Bayes risk, optimal Bayes decisions and confusion matrixes
* `model_tuning.py`: performs further analysis on the selected most promising models (RBF kernel SVM, quadratic logistic regression and GMM with 8 components). In particular, it contains functions to estimate the optimal threshold and apply it to a test set, to calibrate scores and to combine the outputs of different classifiers. All these actions are performed with k-fold cross validation. The main function trains the three selected models and analyses their actual DCF with no calibration, with score calibration and also by estimating an optimal threshold. Then, it also analyses the fusion of the selected models in terms of min DCF and actual DCF. Results are printed on the console.
* `evaluation.py`: contains the functions to evaluate the decisions on the test set. The main function trains many variations of the previous models on the training set and calculates the min DCF on the test set. The tested models are Gaussian, logistic regression, SVM and GMM. Then, the actual DCF of the selected models is evaluated, using also score calibration and optimal threshold estimation. In conclusion, actual DCF of fusions of different models are evaluated, and corresponding ROC plots and Bayes error plots are printed. All results are stored in textual files or in png files.

