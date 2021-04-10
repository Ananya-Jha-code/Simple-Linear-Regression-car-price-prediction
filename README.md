# Simple-Linear-Regression-car-price-prediction
 A simple linear regression model to predict the car price.
 
 The jupyter notebook involves the following:
  1. Importing the necessary libraries using to create a simple linear regression model.
  
  2. Loading a dataset to work on
  3. Making training set

  4. Scaling
  5. Prediction in the form of Array
  6. Visual prediction
 
 
##  1. Importing the necessary libraries to create a simple linear regression model
       import numpy as np  - for linear algebra operations
       import pandas as pd  - for processing data 
       import matplotlib as plt  - for reading the dataset which will be loaded to try the algorithm on.

The dataset can be downloaded from any desired website. 
The website I used for the following project is kaggle.com

##  2. Loading a sample dataset to work on
       data = pd.csv('C:\\Users\\ananya\\laboid\\notebook\\car price\\carpricedata.csv')
   After downloading the dataset of your choice, copy the path of the file and paste it. According to the type of file, change the code. Here my file is in the form of csv.
   You can assign the values of the fields of the dataset to variables as per the purpose of the algorithm. Here the purpose of the algorithm is to predict the car price, where we need to assign the criteria we are looking for as one variable let's say x and a the outcome of it in the form of price as the other variable let's say y.

##  3. Making training set
       from sklearn.model_selection import train_test_split
       x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3,random_state=0)
       
   We split arrays or matrices into random train and test subsets by importing the train_test_split from sklearn.model_selection
   Quick utility that wraps input validation and next(ShuffleSplit().split(X, y)) and application to input data into a single call for splitting (and optionally subsampling) data in a oneliner.    
 
   
 ##  4. Scaling
     from sklearn.preprocessing import StandardScaler
     sc = StandardScaler()

     X_train = sc.fit_transform(x_train)
     X_test = sc.fit_transform(x_test)

     Y_train = sc.fit_transform(y_train)
     Y_test = sc.fit_transform(y_test)
  The sklearn.preprocessing package provides several common utility functions and transformer classes to change raw feature vectors into a representation that is more suitable for the downstream estimators.  
  Standardization of datasets is a common requirement for many machine learning estimators implemented in scikit-learn; they might behave badly if the individual features do not more or less look like standard normally distributed data
##  5. Prediction in the form of an array
     from sklearn.linear_model import LinearRegression
     lr = LinearRegression()
     lr.fit(x_train, y_train)

     pred1 = lr.predict(x_test)
   Linear regression module is imported from sklearn.linear_model and the training set created previously are used as arguments in the lr.fit function. 
   a prediction function is defined as pred1 which stores the outcome of the prediction in the form of array.
   ##  6. Visual prediction
       x_train2=x_train2.sort_values(by="curbweight")
       y_train2=y_train2.sort_index()

       plt.plot(x_train2,y_train2,)
       plt.plot(x_test2,pred2)

       plt.title("car prices")
       plt.xlabel("curbweight")
       plt.ylabel("price")
       
  For this prediction we have taken curbweight as the characteristic to be checked and get the price of a car which matches it. 
    To make the prediction easier to understand we can plot the prediction. X axis consists of the curbweight and Y axis of the price.
    The graph plotted in blue states the actual price of the car corresponding to the curbweight and the graph plotted in yellow states the price of the car predicted.
   
   
      
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
