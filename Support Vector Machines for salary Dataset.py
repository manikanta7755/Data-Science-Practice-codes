#Support vector machine for salary data set
#Loading the packages

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
from sklearn.svm import SVC

#Loading the testing and training data set
salary_train = pd.read_csv("G:\\Mani\\SVM\\SalaryData_Train(1).csv")
salary_test = pd.read_csv("G:\\Mani\\SVM\\SalaryData_Test(1).csv")

salary_train.describe()
salary_test.describe()


#Converting the nonnumeric data to the numeric data
salary_train['workclass']= labelencoder.fit_transform(salary_train['workclass'])
salary_train['education']= labelencoder.fit_transform(salary_train['education'])
salary_train['maritalstatus']= labelencoder.fit_transform(salary_train['maritalstatus'])
salary_train['occupation']= labelencoder.fit_transform(salary_train['occupation'])
salary_train['relationship']= labelencoder.fit_transform(salary_train['relationship'])
salary_train['race']= labelencoder.fit_transform(salary_train['race'])
salary_train['sex']= labelencoder.fit_transform(salary_train['sex'])
salary_train['native']= labelencoder.fit_transform(salary_train['native'])
salary_train['Salary']= labelencoder.fit_transform(salary_train['Salary'])

salary_test['workclass']= labelencoder.fit_transform(salary_test['workclass'])
salary_test['education']= labelencoder.fit_transform(salary_test['education'])
salary_test['maritalstatus']= labelencoder.fit_transform(salary_test['maritalstatus'])
salary_test['occupation']= labelencoder.fit_transform(salary_test['occupation'])
salary_test['relationship']= labelencoder.fit_transform(salary_test['relationship'])
salary_test['race']= labelencoder.fit_transform(salary_test['race'])
salary_test['sex']= labelencoder.fit_transform(salary_test['sex'])
salary_test['native']= labelencoder.fit_transform(salary_test['native'])
salary_test['Salary']= labelencoder.fit_transform(salary_test['Salary'])

#Converted training and testing data input and output
train_X = salary_train.iloc[:, :-1]
train_y = salary_train.iloc[:, -1]
test_X  = salary_test.iloc[:, :-1]
test_y  = salary_test.iloc[:, -1]


#Generating linear kernal model
salary_linear = SVC(kernel = "linear")
salary_linear.fit(train_X, train_y)
pred_test_linear = salary_linear.predict(test_X)

np.mean(pred_test_linear == test_y)

# Generating RBF kernal model
salary_rbf = SVC(kernel = "rbf")
salary_rbf.fit(train_X, train_y)
pred_test_rbf = salary_rbf.predict(test_X)

np.mean(pred_test_rbf==test_y)