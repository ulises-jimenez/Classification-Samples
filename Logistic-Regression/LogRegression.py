import pandas as pd
import statsmodels.api as sm
import numpy as np


data = pd.DataFrame.from_csv('Smarket.csv') #imports main dataset
data['Intercept'] = 1 #sets intercept for regression
train = data[data['Year'] != 2005] #gets training data from years 2001-2004 to predict 2005

trainDir = train['Direction'] # direction of stock movements in training data

del(train['Direction'], train['Today'], train['Year']) # Data cleaning


check = np.where(trainDir == 'Up', 1, 0) # Changing directional string values in integers for the regression


test = data[data['Year'] == 2005] #filtering test data
del(test['Direction'], test['Today'], test['Year']) # cleaning test data for predictions using the model created



logit = sm.Logit(check, train) # creating model using training data

result = logit.fit() #creating fit object

print result.summary() # model summary
preds = result.predict(test) # predictions made from training data on the test data

testresults = np.where(preds > .5, 'Up', 'Down') #numeric predictions converted into Up or Down strings

eval = data[data['Year'] == 2005].reset_index()
del(eval['index'])


dataupDir = eval[eval.Direction == 'Up'] #data filtered by year 2005 and direction up
datadownDir = eval[eval.Direction == 'Down'] #data filtered by year 2005 and direction up

testresults = pd.Series(testresults)

upcomparison = testresults.ix[dataupDir.index] == dataupDir.Direction
downcomparison = testresults.ix[datadownDir.index] == datadownDir.Direction

correct_ups = len(upcomparison[upcomparison == True]) # number of times up was correctly predicted
correct_downs = len(downcomparison[downcomparison == True]) # number of times down was correctly predicted

total_predictions = len(testresults)
correct_prozent = (float(correct_ups+correct_downs)/float(total_predictions))*100
test_up_success_rate = float(correct_ups)/float(len(upcomparison))
test_down_success_rate = float(correct_downs)/float(len(downcomparison))


print "There were %d total predictions" %total_predictions
print "\n%.2f percent of the total predictions were correct" % correct_prozent
print "\n%.2f percent of the predicted Up movements were correct" % test_up_success_rate
print "\n%.2f percent of the predicted Down movements were correct" % test_down_success_rate

