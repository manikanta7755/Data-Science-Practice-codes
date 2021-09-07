## Naive bayes for Tweets data set
#Importing Packages into Python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

#Importing the file into Python
tweets_data = pd.read_csv("G:\\Mani\\naive base\Disaster_tweets_NB.csv",encoding = "ISO-8859-1")

import re
stopword=[]
#Loading the stop words file
with open("C:\Users\User\Downloads\stop.txt", "r") as sw:
    stop_words = sw.read()
    
stop_words = stop_words.split("\n")

#Defining the custom function to clean the data 
def cleaning_text(i):
    i = re.sub("[^A-Za-z" "]+"," ",i).lower()
    i = re.sub("[0-9" "]+"," ",i)
    w = []
    for word in i.split(" "):
        if len(word)>3:
            w.append(word)
    return (" ".join(w))
    
#Applying custom function the main data set for cleaning
tweets_data.text = tweets_data.text.apply(cleaning_text)

# Empty tweets are to be removed if any
email_data = email_data.loc[email_data.text != " ",:]


# Converting the collection of text documents in to a matrix of token counts

# Splitting the tweets data set into train and test data sets 
from sklearn.model_selection import train_test_split

tweets_data_train, tweets_data_test = train_test_split(tweets_data, test_size = 0.2)

# creating a matrix of token counts for the entire tweets text document 
def split_into_words(i):
    return [word for word in i.split(" ")]

# Creating bag of words for the tweets data 
tweets_data_bow = CountVectorizer(analyzer = split_into_words).fit(tweets_data.text)

# Defining Bag of words for all the tweets
all_tweets_matrix = tweets_data_bow.transform(tweets_data.text)

# Defining bag of words for training tweets
train_tweets_matrix = tweets_data_bow.transform(tweets_data_train.text)

# Defining bag of words for testing tweets
test_tweets_matrix = tweets_data_bow.transform(tweets_data_test.text)


# Learning Term weighting and normalizing on entire emails
tfidf_transformer = TfidfTransformer().fit(all_tweets_matrix)

# Preparing TFIDF for training tweets
train_tfidf = tfidf_transformer.transform(train_tweets_matrix)
train_tfidf.shape

# Preparing TFIDF for testing tweets
test_tfidf = tfidf_transformer.transform(test_tweets_matrix)
test_tfidf.shape 


# Preparing a naive bayes model on training tweets data 

from sklearn.naive_bayes import MultinomialNB as MB

# Multinomial Naive Bayes
classifier_mb = MB()
classifier_mb.fit(train_tfidf, tweets_data_train.type)

# Evaluating the testing tweets data
test_prediction = classifier_mb.predict(test_tfidf)
test_prediction = np.mean(test_prediction == tweets_data_test.type)
test_prediction

#Finding the accuracy of the model
from sklearn.metrics import accuracy_score
accuracy_score(test_prediction, tweets_data_test.type) 

pd.crosstab(test_prediction, tweets_data_test.type)

# Finding the accuracy of the training data
train_prediction = classifier_mb.predict(train_tfidf)
train_prediction = np.mean(train_prediction == tweets_data_train.type)
train_prediction