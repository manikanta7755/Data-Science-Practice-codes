# Multilinear Regression for toyoto corollo data set
#Loading the packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
# creating instance for one hot encoding
enc = OneHotEncoder()

from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# loading the data set into Python
toyota = pd.read_csv("G:\\Mani\\Multiple linear regression\\ToyotaCorolla.csv")

toyota.drop(toyota.columns.difference(['Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight']), 1, inplace=True)

toyota.describe()

#Graphical Representation
import matplotlib.pyplot as plt # mostly used for visualization purposes 

# HP
plt.bar(height = toyota.Price, x = np.arange(1, 1437, 1))
plt.hist(toyota.Price) #histogram
plt.boxplot(toyota.Price) #boxplot

# MPG
plt.bar(height = toyota.Price, x = np.arange(1, 6260, 1))
plt.hist(toyota.Price) #histogram
plt.boxplot(toyota.Price) #boxplot


# Q-Q Plot
from scipy import stats
import pylab
stats.probplot(toyota.Price, dist = "norm", plot = pylab)
plt.show()

# Scatter plot between the variables along with histograms
import seaborn as sns
sns.pairplot(toyota.iloc[:, :])
                             
# Correlation matrix 
toyota.corr()

# we see there exists High collinearity between input variables especially between
# [HP & SP], [VOL & WT] so there exists collinearity problem

# preparing model considering all the variables 
import statsmodels.formula.api as smf # for regression model
         
ml1 = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit() # regression model

# Summary
ml1.summary()
# p-values for WT, VOL are more than 0.05

# Checking whether data has any influential values 
# Influence Index Plots
import statsmodels.api as sm

sm.graphics.influence_plot(ml1)
# Studentized Residuals = Residual/standard deviation of residuals
# index 76 is showing high influence so we can exclude that entire row
toyota_new = toyota.drop(toyota.index[[80]])

# Preparing model                  
ml_new = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit()    

# Summary
ml_new.summary()

# Check for Colinearity to decide to remove a variable using VIF
# Assumption: VIF > 10 = colinearity
# calculating VIF's values of independent variables
rsq_Price = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Price = 1/(1 - rsq_Price) 

rsq_Age = smf.ols('Age_08_04 ~ Price + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Age = 1/(1 - rsq_Age)

rsq_KM = smf.ols('KM ~ Age_08_04 + Price + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_KM = 1/(1 - rsq_KM) 

rsq_HP = smf.ols('HP ~ Age_08_04 + KM + Price + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_HP = 1/(1 - rsq_HP) 

rsq_cc = smf.ols('cc ~ Age_08_04 + KM + HP + Price + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_cc = 1/(1 - rsq_cc) 

rsq_Doors = smf.ols('Doors ~ Age_08_04 + KM + HP + cc + Price + Gears + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Doors = 1/(1 - rsq_Doors) 

rsq_Gears = smf.ols('Gears ~ Age_08_04 + KM + HP + cc + Doors + Price + Quarterly_Tax + Weight', data = toyota).fit().rsquared  
vif_Gears = 1/(1 - rsq_Gears) 

rsq_quaterly = smf.ols('Quarterly_Tax ~ Age_08_04 + KM + HP + cc + Doors + Gears + Price + Weight', data = toyota).fit().rsquared  
vif_quaterly = 1/(1 - rsq_quaterly) 

rsq_weight = smf.ols('Weight ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Price', data = toyota).fit().rsquared  
vif_weight = 1/(1 - rsq_weight) 


# Storing vif values in a data frame
d1 = {'Variables':['Price','Age_08_04', 'KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'], 'VIF':[vif_Price, vif_Age, vif_KM, vif_HP,vif_cc, vif_Doors,vif_Gears ,vif_quaterly,vif_weight]}
Vif_frame = pd.DataFrame(d1)  
Vif_frame
# As WT is having highest VIF value, we are going to drop this from the prediction model

# Final model
final_ml = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota).fit()
final_ml.summary() 

# Prediction
pred = final_ml.predict(toyota)

# Q-Q plot
res = final_ml.resid
sm.qqplot(res)
plt.show()

# Q-Q plot
stats.probplot(res, dist = "norm", plot = pylab)
plt.show()

import seaborn as sns
# Residuals vs Fitted plot
sns.residplot(x = pred, y = toyota.Price, lowess = True)
plt.xlabel('Fitted')
plt.ylabel('Residual')
plt.title('Fitted vs Residual')
plt.show()

sm.graphics.influence_plot(final_ml)

### Splitting the data into train and test data 
from sklearn.model_selection import train_test_split
toyota_train, toyota_test = train_test_split(toyota, test_size = 0.2) # 20% test data

# preparing the model on train data 
model_train = smf.ols('Price ~ Age_08_04 + KM + HP + cc + Doors + Gears + Quarterly_Tax + Weight', data = toyota_train).fit()

# prediction on test data set 
test_pred = model_train.predict(toyota_test)

# test residual values 
test_resid = test_pred - toyota_test.Price
# RMSE value for test data 
test_rmse = np.sqrt(np.mean(test_resid * test_resid))
test_rmse

# train_data prediction
train_pred = model_train.predict(toyota_train)

# train residual values 
train_resid  = train_pred - toyota_train.Price
# RMSE value for train data 
train_rmse = np.sqrt(np.mean(train_resid * train_resid))
train_rmse
