##downloads the "testing" and "training" datasets.
# The "testing" dataset is referred to as "training_full" because it
# will be subdivided into new training and testing sets for model building.
# The "training" dataset will be referred to as "validation", because it
# will be used to validate our final model.

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, destfile = "training_full.csv")

url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, destfile = "validation.csv")

##loads training.csv and validation.csv

training_full <- read.csv("training_full.csv", na.strings= c("", " ", "na", "Na", "NA", "#DIV/0!"))
validation <- read.csv("validation.csv", na.strings= c("", " ", "na", "Na", "NA", "#DIV/0!"))

## inspection of the variables for NA values revealed that many variables are composed almost entirely
## of NA's and could therefore be deleted from the data set without hindering out ability to
## predict

na_vars <- sapply(training_full, function(x) {length(which(is.na(x)))})
training_lean <- training_full[ , na_vars < 1000 ]
validation <- validation[ , na_vars < 1000 ]

## loads caret package and splits the "training_full" dataset
## into two subsets, "training", and "testing"

library(caret)

intrain <- createDataPartition(training_lean$classe, p = .60, list = F)
training <- training_lean[intrain, ]
testing <- training_lean[-intrain, ]

#Next we investigated the variables involved

##investigates the "classe" variable (it is a factor variable with 5 levels)
# this is the variable we will attempt to predict

str(training_lean$classe)
summary(training_lean$classe)

# We created a table of the variable "user_name" and discovered that there were 
# six participants in the study. By plotting this against the variable 
# "cvtd_timestamp" we determined that the data was collected over a different
# 2-3 minute time period for each participant. By plotting "classe" against
# "cvtd_timestamp" we also determined that participants performed each "classe"
# in the same order. Since our objective is to predict based on motion data we
# decided to eliminate variables related to the time of an exercise or the person
# performing it (variables 1-5). Variables new_window and num_window also seemed
# to be related to 

table(training$user_name)

qplot(training$user_name, training$cvtd_timestamp)

qplot(training$classe, training$cvtd_timestamp)

qplot(training$classe, training$X)
qplot(training$user_name, training$X)


## eliminates extraneous variables
training <- training[, -c(1:7) ]
testing <- training[, -c(1:7) ]
validation <- validation[ , -c(1:7)]

# We decided to build our first potential prediction model using decision trees.
# Since the outcome variable is categorical, we did not attempt a regression model.
set.seed(91660)

library(rpart)

treeMod <- train(classe ~ ., method = "rpart", data = training,
                 trControl = trainControl(method = "cv", number = 15))

## The predicting with trees produced a model that perfectly identified classe's
## A, B, and E, however it misclassified C and D more often that it correctly identified
## them. I tried.

confusionMatrix(treeMod)
treeMod$finalModel
## I attempted again, this time decreasing the number of folds in the hope that the
## increased sample size would allow for more branches on our tree. This reduced the error.


treeMod2 <- train(classe ~ ., method = "rpart", data = training,
                 trControl = trainControl(method = "cv", number = 7))
confusionMatrix(treeMod2)
treeMod2$finalModel
treeMod2$results

treeMod3 <- train(classe ~ ., method = "rpart", data = training,
                  trControl = trainControl(method = "cv", number = 5))

confusionMatrix(treeMod3)
treeMod3$finalModel
treeMod3$results

# I ultimately decided that the best rpart model would use 5 folds.
# Cross-validating on the testing set showed an out-of-sample accuracy of .66, which was
# far too low for a predictive model.

rtPredict <- predict(treeMod2, testing)
confusionMatrix(rtPredict, testing$classe)

# Next attempt used random forests. The randomForest() function cross-validates internally,
# so no specific parameters needed to be set. The confusion matrix of the model showed perfect prediction
# accuracy. This model was used to predict on the testing set and again showed perfect accuracy.
# 
library(randomForest)

rfMod <- randomForest(classe ~ ., data = training)
rfMod$confusion
frmod$err.rate

rfPredict <- predict(rfMod, testing)
confusionMatrix(rfPredict, testing$classe)