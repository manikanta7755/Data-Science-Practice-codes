### Multilinear Regression for company startuos data set###

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance for ine hot encoding
enc = OneHotEncoder()

# loading the data set into python
startup = pd.read_csv("G:\\Mani\\Multiple linear regression\\50_Startups.csv")

def OneHotEncoders(name,data):
    sample = data[name]
    sample=sample.to_frame()
    enc_sample = pd.DataFrame(enc.fit_transform(sample).toarray())
    enc_sample=enc_sample.iloc[:,:-1]
    data.drop([name],axis=1,inplace=True)
    data=pd.concat([data,enc_sample],axis=1)
    return data

startup=OneHotEncoders("State",startup)

# Exploratory data analysis:
# 1. Measures of central tendency
# 2. Measures of dispersion
# 3. Third moment business decision
# 4. Fourth moment business decision
# 5. Probability distributions of variables 
# 6. Graphical representations (Histogram, Box plot, Dot plot, Stem & Leaf plot, Bar plot, etc.)

startup.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# HP
plt.bar(height = startup["RD"], x = np.arange(1, 51, 1))
plt.hist(startup["RD"]) #histogram
plt.boxplot(startup["RD"]) #boxplot

# MPG
plt.bar(height = startup.Profit, x = np.arange(1, 51, 1))
plt.hist(startup.Profit) #histogram
plt.boxplot(startup.Profit) #boxplot

# Jointplot
import seaborn as sns
sns.jointplot(x=startup['RD'], y=startup['Profit'])

# Countplot
plt.figure(1, figsize=(16, 10))
sns.countplot(startup['RD'])

# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(startup.Profit, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(startup.iloc[:, :])
                             
# Correlation matrix 
startup.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Profit ~ RD + Administration + Marketing + 0 + 1', data = startup).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row

startup_new = startup.drop(startup.index[[49]])

# Preparing model                  
ml_new = smf.ols('Profit ~ RD + Administration + Marketing + 0 + 1', data = startup_new).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_hp = smf.ols('Profit ~ RD + Administration + Marketing', data = startup).fit().rsquared  
vif_hp = 1/(1 - rsq_hp) 

rsq_wt = smf.ols('RD ~ Administration + Marketing', data = startup).fit().rsquared  
vif_wt = 1/(1 - rsq_wt)

rsq_vol = smf.ols('Administration ~ RD + Marketing', data = startup).fit().rsquared  
vif_vol = 1/(1 - rsq_vol) 

rsq_sp = smf.ols('Marketing ~ RD + Administration', data = startup).fit().rsquared  
vif_sp = 1/(1 - rsq_sp) 

# Storing vif values in a data frame
d1 = {'Variables':['Profit','RD', 'Administration', 'Marketing'], 'VIF':[vif_hp, vif_wt, vif_vol, vif_sp]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Profit ~ RD + Administration + Marketing', data = startup).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(startup)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

# Residuals vs Fitted plot
sns.residplot(x = pred, y = startup.Profit, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
startup_train, startup_test = train_test_split(startup, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols("Profit ~ RD + Administration + Marketing", data = startup_train).fit()

# prediction on test data set 
test_pred = model_train.predict(startup_test)

# test residual values 
test_resid = test_pred - startup_test.Profit
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(startup_train)

# train residual values 
train_resid  = train_pred - startup_train.Profit
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
