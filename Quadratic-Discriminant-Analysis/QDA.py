import pandas as pd
import sklearn.qda as qda
import numpy as np
from sklearn import cross_validation
from sklearn import metrics

data = pd.DataFrame.from_csv('Smarket.csv') #imports main dataset

X_train = data[data['Year'] != 2005] #gets training data from years 2001-2004 to predict 2005

y_train = X_train['Direction'] # direction of stock movements in training data

X_train = pd.DataFrame(X_train, columns = ['Lag1', 'Lag2'])

y_train = np.where(y_train == 'Up', 1, 0) # Changing directional string values in integers for the regression

X_test = data[data['Year'] == 2005] #filtering test data
y_test = X_test['Direction']

y_test = np.where(y_test == 'Up', 1, 0) # Changing directional string values in integers for the regression
X_test = pd.DataFrame(X_test, columns = ['Lag1', 'Lag2'])

est = qda.QDA() # creates QDA object
model = est.fit(X_train, y_train) #creates estimator with training data

print model.score(X_test, y_test) #Scores the model on test data

predicted = model.predict(X_test) #gives the predicted y values from the test data predictors

print metrics.classification_report(y_test, predicted) # prints report about the model's performance

print metrics.confusion_matrix(y_test, predicted) # prints confusion matrix

kfold = cross_validation.KFold(len(data), n_folds = 125) # creates K-fold validation

kfmean = np.mean([model.score(X_test, y_test) # estimates the test error by using a kfold cross validation method
	for train, test in kfold])
	
print "\n\nThe Estimated MSE by Cross validation over 125 k-folds, each with a length 10, is %.4f" % kfmean
