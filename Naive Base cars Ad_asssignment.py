# Navie Bayes for cars data set
#Importing packages into python

import pandas as pd
import numpy as np

#Importing the file into python
cars_data = pd.read_csv("G:\\Mani\\naive base\\NB_Car_Ad.csv",encoding = "ISO-8859-1")

#Splitting the data into train and test data
from sklearn.model_selection import train_test_split

cars_data_train, cars_data_test = train_test_split(cars_data, test_size = 0.2)

#Importing navie bayes package
from sklearn.naive_bayes import MultinomialNB as MB

# Applying Navie Bayes model for the data sets
classifier_mb = MB()
classifier_mb.fit(cars_data_train,cars_data_train.Purchased)

#Evaluating with the text data
test_prediction = classifier_mb.predict(cars_data_test)
test_prediction  = np.mean(test_prediction  == cars_data_test.Purchased)
test_prediction 

#Forming a matrix to visualize the prediction
from sklearn.metrics import accuracy_score
accuracy_score(test_prediction, cars_data_test.Purchased) 

pd.crosstab(test_prediction, cars_data_test.Salary)
