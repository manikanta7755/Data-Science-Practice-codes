##Simple Linear Regression for Employee data set##
#importing the packages

import pandas as pd   
import numpy as np 
import matplotlib.pyplot as plt

#Loading the Employee data set into python
employee_data = pd.read_csv("G:\\Mani\\simple linear regression\\emp_data.csv")
employee_data.describe()

#Visualization of barplot, box plot and histograms

plt.bar(height=employee_data.Salary_hike,x = np.arange(1, 110, 1))
plt.hist(height=employee_data.Salary_hike)
plt.boxplot(height=employee_data.Salary_hike)

plt.bar(height=employee_data.Churn_out_rate,x = np.arange(1, 110, 1))
plt.hist(height=employee_data.Churn_out_rate)
plt.boxplot(height=employee_data.Churn_out_rate)

#Scatter Plot
plt.scatter(x = employee_data['Salary_hike'], y = employee_data['Churn_out_rate'], color = 'green') 

# correlation
np.corrcoef(employee_data.Salary_hike, employee_data.Churn_out_rate) 
#Covarience
cov_output = np.cov(employee_data.Salary_hike, employee_data.Churn_out_rate)[0, 1]
cov_output

#Simple linear regression
import statsmodels.formula.api as smf
#Generating the model
model = smf.ols('Churn_out_rate ~ Salary_hike', data = employee_data).fit()
model.summary()

prediction = model.predict(pd.DataFrame(employee_data['Salary_hike']))

# Scatter plot and the regression line
plt.scatter(employee_data.Salary_hike, employee_data.Churn_out_rate)
plt.plot(employee_data.Salary_hike, prediction, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result = employee_data.Churn_out_rate - prediction
res_sqr = result * result
mse = np.mean(res_sqr)
RMSE = np.sqrt(mse)
RMSE

#Applying transformation techniques to prune the model
#Applying logarthemic transformation
plt.scatter(x = np.log(employee_data['Salary_hike']), y = employee_data['Churn_out_rate'], color = 'brown')
#Correlation
np.corrcoef(np.log(employee_data.Salary_hike), employee_data.Churn_out_rate) 

model2 = smf.ols('Churn_out_rate ~ np.log(Salary_hike)', data = employee_data).fit()
model2.summary()

prediction2 = model2.predict(pd.DataFrame(employee_data['Salary_hike']))

# Scatter plot and the regression line
plt.scatter(np.log(employee_data.Salary_hike), employee_data.Churn_out_rate)
plt.plot(np.log(employee_data.Salary_hike), prediction2, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result2 = employee_data.Churn_out_rate - prediction2
res_sqr2 = result2 * result2
mse2 = np.mean(res_sqr2)
RMSE2 = np.sqrt(mse2)
RMSE2

#Applying Exponentian Technique
plt.scatter(x =  employee_data['Salary_hike'], y = np.log(employee_data['Churn_out_rate']), color = 'orange')
np.corrcoef(employee_data.Salary_hike, np.log(employee_data.Churn_out_rate))

model3 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike', data = employee_data).fit()
model3.summary()

prediction3 = model3.predict(pd.DataFrame(employee_data['Salary_hike']))
pred3_at = np.exp(prediction3)
pred3_at

# Scatter plot and the regression line
plt.scatter(employee_data.Salary_hike, np.log(employee_data.Churn_out_rate))
plt.plot(employee_data.Salary_hike, prediction3, "r")
plt.legend(['Predicted line', 'Observed data'])
plt.show()

#Calculating the error
result3 = employee_data.Churn_out_rate - pred3_at
res_sqr3 = result3 * result3
mse3 = np.mean(res_sqr3)
RMSE3 = np.sqrt(mse3)
RMSE3

#Applying polynomial transformation technique
model4 = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = employee_data).fit()
model4.summary()

prediction4 = model4.predict(pd.DataFrame(employee_data))
pred4_at = np.exp(prediction4)
pred4_at

#Scatter plot and the regression line
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 2)
X = employee_data.iloc[:, 0:1].values
X_poly = poly_reg.fit_transform(X)

plt.scatter(employee_data.Salary_hike, np.log(employee_data.Churn_out_rate))
plt.plot(X, prediction4, color = 'red')
plt.legend(['Predicted line', 'Observed data'])
plt.show()

# Calculating the error
result4 = employee_data.Churn_out_rate - pred4_at
res_sqr4 = result4 * result4
mse4 = np.mean(res_sqr4)
RMSE4 = np.sqrt(mse4)
RMSE4

# Choosing the best model using RMSE
data = {"MODEL":pd.Series(["SLR", "Log model", "Exp model", "Poly model"]), "RMSE":pd.Series([RMSE, RMSE2, RMSE3, RMSE4])}
table_rmse = pd.DataFrame(data)
table_rmse

#Selecting the best model for the final modeling of the data set
from sklearn.model_selection import train_test_split

train, test = train_test_split(employee_data, test_size = 0.2)

finalmodel = smf.ols('np.log(Churn_out_rate) ~ Salary_hike + I(Salary_hike*Salary_hike)', data = train).fit()
finalmodel.summary()

# Predict on test data
test_pred = finalmodel.predict(pd.DataFrame(test))
pred_test_Churn_out_rate = np.exp(test_pred)
pred_test_Churn_out_rate

# Model Evaluation on Test data
test_res = test.Churn_out_rate - pred_test_Churn_out_rate
test_sqrs = test_res * test_res
test_mse = np.mean(test_sqrs)
test_rmse = np.sqrt(test_mse)
test_rmse

# Prediction on train data
train_pred = finalmodel.predict(pd.DataFrame(train))
pred_train_Churn_out_rate = np.exp(train_pred)
pred_train_Churn_out_rate

# Model Evaluation on train data
train_res = train.Churn_out_rate - pred_train_Churn_out_rate
train_sqrs = train_res * train_res
train_mse = np.mean(train_sqrs)
train_rmse = np.sqrt(train_mse)
train_rmse