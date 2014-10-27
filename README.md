Classification-Samples
======================

All of the classification samples in this repository are performed on daily stock movement data from 2001 to 2005.
The data can be found in the csv file here. Each model attempts to predict stock price movements using the variables found in the csv file Smarket.csv. The lag variables record the returns for the previous 5 days. The amount of shares traded on the previous day is also included in volume. 

The script located in the Classification-Optimization folder performs all of the methods in this repository on the data at once and recommends the one with the highest crossvalidated kfolds score. This turns out to be the Quadratic Discriminant Analysis method, which correctly predicts the stock movements 60% of the time. Of course this result would be more useful with a larger dataset.



Collection of Classification Samples
