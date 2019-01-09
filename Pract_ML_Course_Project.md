---
title: "Practical Machine Learning Course Project"
author: "Garold Minkin"
date: "January 09, 2019"
output: 
  html_document:
    keep_md: true
---

## Background

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement - a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it.  

In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).  

## Data

The training data for this project are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here:  
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
  
The files have been been downloaded and are assumed to be in the current working directory.  
  


## Exploratory Data Analysis and Preparation  

### Data Load  


```r
setwd("c:/work/data_science/Course8_ML/project")
# -- while loading, convert empty or irregular values to NA
train_set <- read.csv('pml-training.csv', header=T, na.strings = c("NA","NaN",""," ","#DIV/0!"))
test_set  <- read.csv('pml-testing.csv', header=T, na.strings = c("NA","NaN",""," ","#DIV/0!"))
```

Training set size is 19622 rows and 160 columns  
Testing set size is 20 rows and 160 columns  

### Data Cleanup  


```r
# -- notice that the 1st 7 columns are irrelevant to the task at hand, so we can remove them  
train_set <- train_set[,-c(1:7)]
test_set  <- test_set [,-c(1:7)]
# -- also notice that many columns contain NA or no data, so we can remove them as well  
train_set <- train_set[, colSums(is.na(train_set)) == 0]
test_set  <- test_set [, colSums(is.na(test_set )) == 0]
```

After cleanup, training set size is 19622 rows and 53 columns  
After cleanup, testing set size is 20 rows and 53 columns  

### Data Partitioning  

For cross-validation, we will further split the training set into training subset and validation subset, apportioned 75% and 25% respectively.  Also, we set the seed to 12345 for reproducability.  


```r
set.seed(12345)
inTrain <- createDataPartition(train_set$classe, p = 0.75, list = FALSE)
train_subset <- train_set[ inTrain, ]
valid_subset <- train_set[-inTrain, ]
```

After partition, training subset size is 14718 rows and 53 columns  
After partition, validation subset size is 4904 rows and 53 columns  

## Prediction Models  

Here we will attempt to use the following 3 different models to predict the _classe_ variable as outcome, using the rest of the variables as predictors:

* Decision Tree
* Random Forest
* Gradient Boosting Model

### Decision Tree Model  


```r
# -- Train the Decision Tree Model using the training subset
model_dt <- train(classe ~ ., data = train_subset, method="rpart")

# -- Plot of the Decision Tree as a dendogram
fancyRpartPlot(model_dt$finalModel, main = "Decision Tree Plot\n", sub = "Predictions", type = 1)
```

![](Pract_ML_Course_Project_files/figure-html/dt_model-1.png)<!-- -->

```r
# -- Generate Predictions using the validation subset:
pred_dt <- predict(model_dt, newdata = valid_subset)

# -- Test results on the validation subset, using the confusion matrix:
cm_dt <- confusionMatrix(pred_dt, valid_subset$classe)
```

Decision Tree Model Accuracy = 0.5415987 (95% Confidence Interval: (0.5275299, 0.5556179))
As we see, this accuracy is rather low and so we cannot predict the _classe_ outcome well by the other predictors using this model.  

### Random Forest Model  


```r
# -- Train the Random Forest Model using the training subset
model_rf <- randomForest(classe ~ ., data = train_subset, method="class")

# -- Generate Predictions using the validation subset:
pred_rf <- predict(model_rf, newdata = valid_subset)

# -- Test results on the validation subset, using the confusion matrix:
cm_rf <- confusionMatrix(pred_rf, valid_subset$classe)
```

Random Forest Model Accuracy = 0.9942904 (95% Confidence Interval: (0.9917585, 0.9962027))

### Gradient Boosting Model  


```r
# -- Train the Gradient Boosting Model using the training subset
control_gb <- trainControl(method = "repeatedcv", number = 4, repeats = 1)
model_gb  <- train(classe ~ ., data = train_subset, method = "gbm", trControl = control_gb, verbose = FALSE)

# -- Generate Predictions using the validation subset:
pred_gb <- predict(model_gb, newdata = valid_subset)

# -- Test results on the validation subset, using the confusion matrix:
cm_gb <- confusionMatrix(pred_gb, valid_subset$classe)
```

Gradient Boosting Model Accuracy = 0.9604405 (95% Confidence Interval: (0.9546021, 0.9657222))

### Prediction Model Selection  

Based on the above analysis, we choose the Random Forest Model as it offers the highest level of accuracy: 0.9942904, or 99.43%  
Out of sample error can be derived as (1-Accuracy) = 0.0057096, or 0.57%  

### Applying Random Forest Model to the test set  

We now attempt to predict the outcome (_classe_ variable) on the test set using the Random Forest Model described above.  
The results are stored as a text file to be uploaded into the github repository.  


```r
# -- Generate outcome predictions using the test data set:
pred_rf_test <- predict(model_rf, newdata = test_set)
print(pred_rf_test)
```

```
##  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 
##  B  A  B  A  A  E  D  B  A  A  B  C  B  A  E  E  A  B  B  B 
## Levels: A B C D E
```

```r
write.table(data.frame(pred_rf_test), file = "outcome_predictions.txt", quote = F, col.names = F)
```
