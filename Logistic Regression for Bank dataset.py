###Logistic linear regression for bank data set###
#Loading the packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import statsmodels.formula.api as sm

from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report

#Loading the data set
bank_data = pd.read_csv("G:\\Mani\\Logistic Regression\\bank_data.csv", sep = ",")

bank_data.describe()
#Identifying the missing values
bank_data.isna().sum()
#Building the basic logistic model
logistic_model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + Pdays + Previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin. + joblue.collar + joentrepreneur + johousemaid + jomanagement + joretired + joself.employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data).fit()

#summary of the model
logistic_model.summary2()
logistic_model.summary()

#Prediction of the model
prediction = logistic_model.predict(bank_data.iloc[ :, 1: ])
#Finding fpr, tpr and threshold values
fpr, tpr, thresholds = roc_curve(bank_data.y, prediction)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_threshold

import pylab as pl

i = np.arange(len(tpr))
roc = pd.DataFrame({'fpr' : pd.Series(fpr, index=i),'tpr' : pd.Series(tpr, index = i), '1-fpr' : pd.Series(1-fpr, index = i), 'tf' : pd.Series(tpr - (1-fpr), index = i), 'thresholds' : pd.Series(thresholds, index = i)})
roc.iloc[(roc.tf-0).abs().argsort()[:1]]

# Plotting tpr vs 1-fpr
fig, ax = pl.subplots()
pl.plot(roc['tpr'], color = 'red')
pl.plot(roc['1-fpr'], color = 'blue')
pl.xlabel('1-False Positive Rate')
pl.ylabel('True Positive Rate')
pl.title('Receiver operating characteristic')
ax.set_xticklabels([])

roc_auc = auc(fpr, tpr)
print("Area under the ROC curve : %f" % roc_auc)

# creating the prediction column and filling with zeros
bank_data["predictions"] = np.zeros(45211)
# taking threshold value and above the prob value will be treated as correct value 
bank_data.loc[prediction > optimal_threshold, "predictions"] = 1
# classification report
classification = classification_report(bank_data["prediction"], bank_data["y"])
classification

#Splitting the data inti training and testing data 
train_data, test_data = train_test_split(bank_data, test_size = 0.2)

# Model building 
# import statsmodels.formula.api as sm
model = sm.logit('y ~ age + default + balance + housing + loan + duration + campaign + Pdays + Previous + poutfailure + poutother + poutsuccess + poutunknown + con_cellular + con_telephone + con_unknown + divorced + married + single + joadmin. + joblue.collar + joentrepreneur + johousemaid + jomanagement + joretired + joself.employed + joservices + jostudent + jotechnician + jounemployed + jounknown', data = bank_data).fit()

#summerising the model 
model.summary2()
model.summary()

# Prediction on Test data set
test_pred = logistic_model.predict(test_data)

# Creating new column for storing predicted class of Attorney
# filling all the cells with zeroes
test_data["test_pred"] = np.zeros(9043)

# taking threshold value as 'optimal_threshold' and above the thresold prob value will be treated as 1 
test_data.loc[test_pred > optimal_threshold, "test_pred"] = 1

# confusion matrix 
confusion_matrix = pd.crosstab(test_data.test_pred, test_data['y'])
confusion_matrix

accuracy_test = (7780 + 361)/(9043) 
accuracy_test

# classification report
classification_test = classification_report(test_data["test_pred"], test_data["y"])
classification_test

#ROC CURVE AND AUC
fpr, tpr, threshold = metrics.roc_curve(test_data["y"], test_pred)

#PLOT OF ROC
plt.plot(fpr, tpr);plt.xlabel("False positive rate");plt.ylabel("True positive rate")

roc_auc_test = metrics.auc(fpr, tpr)
roc_auc_test
# prediction on train data
train_pred = model.predict(train_data.iloc[ :, 1: ])

# Creating new column 
# filling all the cells with zeroes
train_data["train_pred"] = np.zeros(36168)

# taking threshold value and above the prob value will be treated as correct value 
train_data.loc[train_pred > optimal_threshold, "train_pred"] = 1

# confusion matrix
confusion_matrx = pd.crosstab(train_data.train_pred, train_data['y'])
confusion_matrx

accuracy_train = (31121 + 1452)/(36168)
print(accuracy_train)

