### k means clustering for Airlines dataset###
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from sklearn.cluster import	KMeans
# from scipy.spatial.distance import cdist 

# Kmeans on Airlines Data set 
Airlines = pd.read_excel("G:\\Mani\\K means clustering\\EastWestAirlines (1).xlsx")

Airlines.describe()

# Normalization function 
def norm_func(i):
    x = (i - i.min())	/ (i.max() - i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(Airlines.iloc[:, 1:])

###### scree plot or elbow curve ############
TWSS = []
k = list(range(2, 9))

for i in k:
    kmeans = KMeans(n_clusters = i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)
    
TWSS
# Scree plot 
plt.plot(k, TWSS, 'ro-');plt.xlabel("No_of_Clusters");plt.ylabel("total_within_SS")

# Selecting 5 clusters from the above scree plot which is the optimum number of clusters 
model = KMeans(n_clusters = 5)
model.fit(df_norm)

model.labels_ # getting the labels of clusters assigned to each row 
mb = pd.Series(model.labels_)  # converting numpy array into pandas series object 
Airlines['clust'] = mb # creating a  new column and assigning it to new column 

Airlines.head()
df_norm.head()

Airlines = Airlines.iloc[:,[11,0,1,2,3,4,5,6,7,8,9,10]]
Airlines.head()

Airlines.iloc[:, 2:8].groupby(Airlines.clust).mean()

Airlines.to_csv("Kmeans_Airlines.csv", encoding = "utf-8")

import os
os.getcwd()
