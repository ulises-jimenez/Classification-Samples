import pandas as pd
import sklearn.lda as lda
import sklearn.qda as qda
import sklearn.linear_model as lm
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import cross_validation
from sklearn import metrics  # imports

data = pd.DataFrame.from_csv('Smarket.csv') #imports main dataset

X_train = data[data['Year'] != 2005] #gets training data from years 2001-2004 to predict 2005

y_train = X_train['Direction'] # direction of stock movements in training data

X_train = pd.DataFrame(X_train, columns = ['Lag1', 'Lag2'])

y_train = np.where(y_train == 'Up', 1, 0) # Changing directional string values in integers for the regression

X_test = data[data['Year'] == 2005] #filtering test data
y_test = X_test['Direction']

y_test = np.where(y_test == 'Up', 1, 0) # Changing directional string values in integers for the regression
X_test = pd.DataFrame(X_test, columns = ['Lag1', 'Lag2'])

input = (X_train, y_train, X_test, y_test) # test and training data subsets for classification models

def picker(cross_val_list): # Uses cross-validated scores as input, then selects the model that scored the best
	scores = cross_val_list
	largest = float(max(scores[:, 0]))
	
	for i in scores:
		
		if largest == float(i[0]):
			name = i[1]
		else:
			pass

	print "\n\n\nThe %s was the most successful classifier with success rate of %.4f, \n\n%s is recommended for use" % (name, largest, name)	


def LDA_analyser(train_predictors, train_responses,test_predictors, test_responses): # runs a LDA classifier on the data, returns its cross-validated score
	
	est = lda.LDA()
	model = est.fit(train_predictors, train_responses)
	
	predicted_y = model.predict(test_predictors)
	
	m_score = model.score(test_predictors, test_responses)
	
	report = metrics.classification_report(test_responses, predicted_y)
	
	kfold = cross_validation.KFold(len(train_predictors)+len(test_predictors), n_folds = 125)
	
	kfmean = np.mean([model.score(test_predictors, test_responses)
	for train, test in kfold])
	
	type = "LDA classifier"
	
	print "\n\nThe Estimated MSE by Cross validation over 125 k-folds, each with a length 10, for the %s is %.4f" % (type, kfmean)
	
	return kfmean, type

def QDA_analyser(train_predictors, train_responses,test_predictors, test_responses): # runs a QDA classifier on the data, returns its cross-validated score
	
	est = qda.QDA()
	model = est.fit(train_predictors, train_responses)
	
	predicted_y = model.predict(test_predictors)
	
	m_score = model.score(test_predictors, test_responses)
	
	report = metrics.classification_report(test_responses, predicted_y)
	
	kfold = cross_validation.KFold(len(train_predictors)+len(test_predictors), n_folds = 125)
	
	kfmean = np.mean([model.score(test_predictors, test_responses)
	for train, test in kfold])
	
	type = "QDA classifier"
	
	print "\n\nThe Estimated MSE by Cross validation over 125 k-folds, each with a length 10, for the %s is %.4f" % (type, kfmean)
	
	return kfmean, type
	
def KNN_classifier(train_predictors, train_responses,test_predictors, test_responses, neighbors): # runs a KNN classifier on the data, returns its cross-validated score. 
																								#The nieghbors parameter can be adjusted to desired magnitude
	
	est = KNeighborsClassifier(n_neighbors = neighbors)
	
	model = est.fit(train_predictors, train_responses)
	
	predicted_y = model.predict(test_predictors)
	
	m_score = model.score(test_predictors, test_responses)
	
	report = metrics.classification_report(test_responses, predicted_y)
	
	kfold = cross_validation.KFold(len(train_predictors)+len(test_predictors), n_folds = 125)
	
	kfmean = np.mean([model.score(test_predictors, test_responses)
	for train, test in kfold])
	
	type = "KNN Classifier %d neighbors" % neighbors
	
	print "\n\nThe Estimated MSE by Cross validation over 125 k-folds, each with a length 10, for the %s is %.4f \n\nThe parameter n was set to %d" % (type, kfmean, neighbors)
	
	return kfmean, type, 

def logistic_regression(train_predictors, train_responses,test_predictors, test_responses): # runs a Logistic Regression classifier on the data, returns its cross-validated score
	
	est = lm.LogisticRegression()
	
	model = est.fit(train_predictors, train_responses)
	
	predicted_y = model.predict(test_predictors)
	
	m_score = model.score(test_predictors, test_responses)
	
	report = metrics.classification_report(test_responses, predicted_y)
	
	kfold = cross_validation.KFold(len(train_predictors)+len(test_predictors), n_folds = 125)
	
	kfmean = np.mean([model.score(test_predictors, test_responses)
	for train, test in kfold])
	
	type = "Logistic Regression Classifier"
	
	print "\n\nThe Estimated MSE by Cross validation over 125 k-folds, each with a length 10, for the %s is %.4f" % (type, kfmean)
	
	return kfmean, type	

a = LDA_analyser(X_train, y_train, X_test, y_test) 
b = QDA_analyser(X_train, y_train, X_test, y_test)
c = KNN_classifier(X_train, y_train, X_test, y_test, 1)
d = KNN_classifier(X_train, y_train, X_test, y_test, 4)
e = logistic_regression(X_train, y_train, X_test, y_test)
lumped = [a,b,c, e]
lumped = np.array(lumped)

picker(lumped)
	
