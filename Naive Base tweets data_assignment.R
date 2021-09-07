#Naive Bayes for tweets data Set.

#Installing Packages

install.packages("readr")
library(readr)

install.packages("e1071")
library(e1071)

install.packages("gmodels")
library(gmodels)

install.packages("tm")
library(tm)

#Loading the tweets file into R
tweets_data<-read.csv(file.choose())

View(tweets_data)
str(tweets_data)

#Converting the labelled data into factors
tweets_data$target<-factor(tweets_data$target)

#Separating the data into training and test data sets
set.seed(1) 
train_index<-sample(c(0,1),replace=T,prob=c(0.2,0.8),nrow(tweets_data))

# Creating  test and train corpus from the tweets data
tweets_corpus_train <- VCorpus(VectorSource(tweets_data$text[train_index==1]))
tweets_corpus_test <- VCorpus(VectorSource(tweets_data$text[train_index==0]))


#Separating the labels for the test and training data
y_train=tweets_data$target[train_index==1]
y_test=tweets_data$target[train_index==0]

# Text processing of the data
tweets_corpus_clean_train <- tweets_corpus_train 
tm_map(content_transformer(tolower))
tm_map(removeNumbers)
tm_map(removeWords, stopwords())
tm_map(removePunctuation) 
tm_map(stemDocument)
tm_map(stripWhitespace)

tweets_corpus_clean_test <- tweets_corpus_test 
tm_map(content_transformer(tolower))
tm_map(removeNumbers) 
tm_map(removeWords, stopwords()) 
tm_map(removePunctuation) 
tm_map(stemDocument) 
tm_map(stripWhitespace)

# Print some tweets after processing
for(i in 1:3){
  print(as.character(tweets_corpus_clean_train [[i]]))
}

#Creating Navie Bayes Model on the data set
tweets_dtm_train<- DocumentTermMatrix(tweets_corpus_clean_train)
tweets_dtm_test<- DocumentTermMatrix(tweets_corpus_clean_test)

# Select words with frequency greater than 5
tweets_dtm_freq_train <- tweets_dtm_train 
findFreqTerms(5) 
tweets_dtm_train[ , .]

tweets_dtm_freq_test <- tweets_dtm_test 
findFreqTerms(5)
tweets_dtm_test[ , .]

# Function to convert numeric in yes or no
convert_counts <- function(x) {
  x <- ifelse(x > 0, "Yes", "No")
}

## Convert the matrices in yes or no
tweets_train <- apply(tweets_dtm_freq_train, MARGIN = 2,convert_counts)

tweets_test <- apply(tweets_dtm_freq_test, MARGIN = 2, convert_counts)

# Trainning the model
tweets_classifier <- naiveBayes(tweets_train, tweets_data$target[train_index==1],laplace = 1)

# Predicting the tweets for the test data. 
tweets_predict <- predict(tweets_classifier,tweets_test)



confusion_matrix<-CrossTable(tweets_predict, tweets_data$target[train_index==0], prop.chisq = FALSE, chisq = FALSE, 
                             prop.t = FALSE,
                             dnn = c("Predicted", "Actual"))

conf_mtx <- table(tweets_predict, tweets_data$target[train_index==0])

print(conf_mtx)