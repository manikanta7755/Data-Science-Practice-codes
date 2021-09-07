##Naive Bayes for Salary Data Sets.

#Installing Packages

install.packages("readr")
library(readr)

install.packages("e1071")
library(e1071)

install.packages("gmodels")
library(gmodels)

#Loading the training and testing files into R
Salary_train<-read.csv(file.choose())
salary_test<-read.csv(file.choose())

View(Salary_train)
View(salary_test)

str(Salary_train)
str(salary_test)

#Converting the non Labelled data to factor data
Salary_train$educationno <- as.factor(Salary_train$educationno)
class(Salary_train)

#Converting the non Labelled data to factor data
salary_test$educationno <- as.factor(salary_test$educationno)
class(salary_test)

#Creating a model using Naive Bayes Function
Model <- naiveBayes(Salary_train$Salary ~ ., data = Salary_train)
Model

#Creating a prediction model
Model_predict <- predict(Model,salary_test)

#Finding the acuracy of the predicted model
mean(Model_predict==salary_test$Salary)

#Assigning the predicted model into Matrix form to visualize 
Crosstable(Model_pred, salary_test$Salary,
           prop.chisq = FALSE, prop.t = FALSE, prop.r = FALSE,
           dnn = c('predicted', 'actual'))