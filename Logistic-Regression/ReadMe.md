The Logistic Regression file contains an anlaysis of stock market data


The Logistic Regression files attempt to predict  stock price movements using the variables found in the csv file Smarket.csv
The stock prices are recorded daily for 5 years. The lag variables record the returns for the previous 5 days. The amount of shares traded on the previous day is also included in volume. The logistic regression model takes in these variables and calculates the probability of an upward stock movement. If this probability is larger than .5 then the next days stock movement is predicted as 'Up'. Here are the results

1. In LogisticRegression1.py the direction is regressed on the previous 5 days stock movements and the volume traded. The training data used were the data from 2001 to 2004. The test data used were the returns from 2005. 

  From Output1.txt here we see that the model predicted correctly only 48% of the time. Theoretically this is worse than randomly guessing. In the summary none of the p-values are signicant but it may still help to take the most significant p values and regress using only those.

2. In Output2.txt only the most significant variables Lag1 and Lag2 were kept. Here the model does significantly better, correctly guessing 56% of the time. More specifically you can see that the upward movement guesses were particularly correct, having a figure of 58%. Trading for upward movements on these days would be the most sensible approach. Of course a more accurate picture could be obtained by having a larger training set. The 58% could simply have happened by chance.
