# Wine-Quality-Detection

### Steps to follow
* Perform proper data visualization: already present plots, identify outliers (maybe box plots)
* Try out all the imported models, paying attention to all possible hyperparameters and improvements
* Use different evaluation techniques

### Various notes
* Support Vector Machines have quite a bad performance, find a way to improve them
* GMM is the best performing model so far
* LDA performs bad as a technique for classification
* try to whiten covariance matrix (?)
* for SVM, maybe try tuning hyperparameter csi?
* quadratic kernel SVM are MUCH WORSE than quadratic logistic regression, WHY???
* RBF kernel SVM works very well, maybe try more values of C and gamma if there is some time to waste :)
* REMEMBER TO CHECK IF CENTERED FEATURES MAKE ANY DIFFERENCE!!:
    * for Gaussian models, no difference if features are z-normalized or not
    * for Linear Regression, not very helpful

### TODO
* ROC plots and minDCF plots
* Check if logistic regression to combine scores is correct (weird results in model_tuning)
* Maybe redo figures of GMMs




