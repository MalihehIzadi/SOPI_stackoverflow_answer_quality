#**********************SVM***********************
library(gdata)
library(e1071)
library(caret)

doSvmLearning <- function(first, last, testCount) {
  set.seed(7)
  allData <- read.csv("so-data.csv")
  allData$Label <- ifelse(allData$Label==1, "     YES     ", "     NO     ")
  allData$Label <- factor(allData$Label)
  test <- sample(first:last, testCount)
  testData <- allData[test, ]
  trainData <- allData[-test, ]
  modFit <- svm(Label~AnswerCount+SSVC+ViewCount+SumAnswersScores+HasAcceptedAnswer+AvgAnswerersReputation, data=trainData, kernel  = "polynomial",
                method="class", gamma=0.03125, cost=262144)# According to the output of tune.svm that is commented below
  pre <-predict(modFit, newdata = testData, type="class")
  confMatrix <- confusionMatrix(pre, testData$Label, positive = "     YES     ")
  confMatrix
}

#tmodel <- tune.svm(Label~AnswerCount+SSVC+ViewCount+SumAnswersScores+HasAcceptedAnswer+AvgAnswerersReputation, data=trainData, 
#                   gamma=2^(-15:-1), cost=2^(0:30), tunecontrol = tune.control(cross=2460))# if you want to run this, at first set trainData

#*************************Neural Network*********************

library(gdata)
library(e1071)
library(nnet)
library(caret)


doNNet <- function(first, last, testCount) {
  set.seed(7)
  allData <- read.csv("so-data.csv")
  allData$Label <- ifelse(allData$Label==1, "     YES     ", "     NO     ")
  allData$Label <- factor(allData$Label)
  test <- sample(first:last, testCount)
  testData <- allData[test, ]
  trainData <- allData[-test, ]
  fitControl <- trainControl(method = "LOOCV", 
                             classProbs = TRUE, 
                             summaryFunction = twoClassSummary)
  nnetGrid <-  expand.grid(size = seq(from = 1, to = 10, by = 1), decay = seq(from = 0.1, to = 0.5, by = 0.1))
  modFit <- nnet(Label~SSVC+ViewCount+SumAnswersScores+HasAcceptedAnswer, data=trainData, trControl = fitControl, tuneGrid = nnetGrid,
                 size=66, maxit = 4000, method="class") # According to the output of tune.nnet that is commented below
  pre <-predict(modFit, newdata = testData, type="class")
  pre <- factor(pre)
  confMatrix <- confusionMatrix(pre, testData$Label, positive = "     YES     ")
  confMatrix
}

#tmodel = tune.nnet(Label~SSVC+ViewCount+SumAnswersScores+HasAcceptedAnswer,
#                   trControl = fitControl, tuneGrid = nnetGrid, data = trainData, maxit = 4000, size = 1:100)# if you want to run this, at first set trainData


#****************************Decision Tree***************

library(gdata)
library(rpart)
library(rpart.plot)
library(caret)
library(RColorBrewer)
library(rattle)

doDTreeLearning <- function(first, last, testCount) {
  set.seed(7)
  allData <- read.csv('so-data.csv')
  allData$Label <- ifelse(allData$Label==1, "     YES     ", "     NO     ")
  allData$Label <- factor(allData$Label)
  test <- sample(first:last, testCount)
  testData <- allData[test, ]
  trainData <- allData[-test, ]
  fitControl <- rpart.control(cp = 0.012, xval = 2460)# cp has been set according to plotcp(cp = 0.005) that is commented below
  modFit <- rpart(Label~AnswerCount+SSVC+ViewCount+SumAnswersScores+HasAcceptedAnswer, data=trainData, control = fitControl, method="class")
  rpart.plot(modFit)
  pre <-predict(modFit, newdata = testData, type="class")
  confMatrix <- confusionMatrix(pre, testData$Label, positive = "     YES     ")
  confMatrix
}

#plotcp(modFit)# if you want to run this, at first set modFit


#***********************Sample Call*******************

doDTreeLearning(1, 3075, 615)


#***********************Feature Selection****************

set.seed(7)
allData <- read.csv('so-data.csv')
allData$Label <- ifelse(allData$Label==1, "needing", "notneeding")
allData$Label <- factor(allData$Label)
test <- sample(first:last, testCount)
testData <- allData[test, ]
trainData <- allData[-test, ]

library(Hmisc)
library(caret)

ctrl <- rfeControl(functions = lmFuncs,
                   method = "repeatedcv",
                   repeats = 1,
                   verbose = FALSE)

lmProfile <- rfe(trainData[, c("AnswerCount", "SSVC", "Score", "SumAnswersScores", "ViewCount", "AvgCommentCount",
                               "AvgAnswerersReputation", "AskerReputation", "FavoriteCount", "CommentCount", "HasAcceptedAnswer")],
                 as.vector(trainData[, c("Label")]),
                 sizes = 5:7,
                 metric = "Accuracy",
                 rfeControl = ctrl, method = "rpart")

ctrl <- gafsControl(functions = caretGA)
obj <- gafs(trainData[, c("AnswerCount", "SSVC", "Score", "SumAnswersScores", "ViewCount", "AvgCommentCount",
                          "AvgAnswerersReputation", "AskerReputation", "FavoriteCount", "CommentCount", "HasAcceptedAnswer")],
            as.vector(trainData[, c("Label")]),
            gafsControl = ctrl, method = "rpart")

ctrl <- safsControl(functions = caretSA)
obj <- safs(trainData[, c("AnswerCount", "SSVC", "Score", "SumAnswersScores", "ViewCount", "AvgCommentCount",
                          "AvgAnswerersReputation", "AskerReputation", "FavoriteCount", "CommentCount", "HasAcceptedAnswer")],
            as.vector(trainData[, c("Label")]),
            safsControl = ctrl, method = "rpart")
