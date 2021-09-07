import pandas as pd
import matplotlib.pylab as plt

airlines = pd.read_excel("G:\\Mani\\H Clustering\\EastWestAirlines.xlsx")

airlines.describe()
airlines.info()


# Normalization function 
def norm_func(i):
    x = (i-i.min())	/ (i.max()-i.min())
    return (x)

# Normalized data frame (considering the numerical part of data)
df_norm = norm_func(airlines.iloc[:, 0:])
df_norm.describe()

# for creating dendrogram 
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch 

z = linkage(df_norm, method = "complete", metric = "euclidean")

# Dendrogram
plt.figure(figsize=(20, 6));plt.title('Hierarchical Clustering Dendrogram');plt.xlabel('Index');plt.ylabel('Distance')
sch.dendrogram(z,)
plt.show()


# Now applying AgglomerativeClustering choosing 5 as clusters from the above dendrogram
from sklearn.cluster import AgglomerativeClustering

h_complete = AgglomerativeClustering(n_clusters = 4, linkage = 'complete', affinity = "euclidean").fit(df_norm) 
h_complete.labels_

cluster_labels = pd.Series(h_complete.labels_)

airlines['clust'] = cluster_labels # creating a new column and assigning it to new column 

airlines = airlines.iloc[:, [12,0,1,2,3,4,5,6,7,8,9,10,11]]
airlines.head()

# Aggregate mean of each cluster
airlines.iloc[:, 1:].groupby(airlines.clust).mean()

# creating a csv file 
airlines.to_csv("airlines output.csv", encoding = "utf-8")

import os
os.getcwd()
