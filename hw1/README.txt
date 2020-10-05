Yifan Zhang
5191471
zhan4372@umn.edu

To run the whole programming problems on CSE labs, you should upgrade scikit-learn to latest version and type command "python3 main.py" in terminal.
By excepting this command, the performance of each algorithm on each data is printed on terminal and 3 images called "digits.png", "boston50.png" and "boston75.png" will be generated and saved to current working directory.
Each image contains the performance with error bars  of a dataset with logistic regression and GNB algorithm.

All data are initialized in "main.py".
To initialize data manually, import "main.py" and
boston50X, boston50Y = getData.getData.bostonPercentile(50)
boston75X, boston75Y = getData.getData.bostonPercentile(75)
DigitsX, digitsY = getData.getDigits()

problem 3:
	To run LDA1dThres(num_crossval), please import "main.py" and execute "main.LDA1dThres(num_crossval)" in python.
	num_crossval should be greater than 1.
	To apply 2-class LDA, import "LDA1dThres.py", and train model using "LDA1dThres.LDA1dThres().fit(X_train, y_train)", predict using "LDA1dThres.LDA1dThres().fit(X_train, y_train).predict(X_test)"

	To run LDA2dGaussGM(num_crossval), please import "main.py" and execute "main.LDA2dGaussGM(num_crossval)" in python.
	To apply LDA projection and bivariate gaussian generative model, import "LDA2dGaussGM.py", and train model using "LDA2dGaussGM.LDA2dGaussGM(num_classes, num_features_to_be_reduced_to).fit(X_train, y_train)", predict using "LDA2dGaussGM.LDA2dGaussGM(num_classes, num_features_to_be_reduced_to).fit(X_train, y_train).predict(X_train)"
	num_crossval should be greater than 1.
	For digits dataset in this homework, num_classes is 10 and num_features_to_be_reduced_to is 2.

problem 4:
	To run logisticRegression(num_splits, train_percent), please import "main.py" and execute "main.logisticRegression(num_splits, train_percent)" in python.
	This function will return the mean error and standard deviation of errors for each percent of the training data set.
	To apply logistic regression, import "logisticRegression.py", and train model using "logisticRegression.logisticRegression(num_classes, num_features, step_size, max_iter, tol).train(X_train, y_train)", and predict using "logisticRegression.logisticRegression(num_classes, num_features, step_size, max_iter, tol).predict(X_test)"
	The default values of parameters:
		step_size: 0.01
		max_iter: 2000
		tol = 0.001
	The tol is the tolerance of the cross entropy error, if the cross entropy error is less than tol, then the error curve is believed to be converged.
	num_crossval should be greater than 1.
	For digits dataset in this homework, num_classes is 10 and num_features_to_be_reduced_to is 64.

	To run naiveBayesGaussian(num_splits, train_percentage), please import "main.py" and execute "main.naiveBayesGaussian(num_splits, train_percentage)" in python.
	This function will return the mean error and standard deviation of errors for each percent of the training data set.
	To apply Naive-Bayes with marginal Gaussian distributions, import "naiveBayesGaussian.py" and train the model using "naiveBayesGaussian.naiveBayesGaussian(num_classes).train(X_train, y_train)", predict using "naiveBayesGaussian.naiveBayesGaussian(num_classes).train(X_train, y_train).predict(X_test)".
	num_crossval should be greater than 1.
	For digits dataset in this homework, num_classes is 10.

For each algorithm class in my code, you can have train a model on same training dataset and use this model to predict on different dataset.
For example:
	LR_obj = logisticRegression.logisticRegression(num_classes, num_features, step_size, max_iter, tol).train(X_train, y_train)
	LR_obj.predict(X_test1)
	LR_obj.predict(X_test2)
	LR_obj.predict(X_test3)
	...

Cross validation:
	To run k-fold cross-validation, please import "main.py" and execute "main.my_cross_val(method, X, y, k)" in python.
	method: an object of any class of above algorithms.
	For instance: method = LDA1dThres()
	X, y: dataset and target set
	k: number of folds
	return errorRate_test, errorRate_train, std_test, std_train, mean_test, mean_train

	To run train test split cross validation, please import "main.py" and execute "main.my_train_test(method, X, y, k, num_splits, train_percent, cross_entropy = False)"
	method: an object of any class of above algorithms.
	For instance: method = LDA1dThres()
	X, y: dataset and target set
	k: number of folds
	num_splits: percentage of the training data, in this homework 0.8
	train_percent: a list of percentages, in this homework [10, 25, 50, 75, 100]
	cross_entropy: default is false, should be true when apply to logistic regression
	return errorRate_test, std_test, mean_test

Acknowledge
Yiwen Xu
Jun-Jee Chao
