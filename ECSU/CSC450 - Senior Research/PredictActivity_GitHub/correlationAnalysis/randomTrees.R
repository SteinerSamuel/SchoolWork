# this file is for the machine learning model, using random forests
# Loading in libraries
library(InformationValue)
library(readr)
library(randomForest)
library(caret)
library(stringr)
library(gridExtra)
rm(list = ls()) ## this will delete all objects in the workspace

#Fscore function
f1score <- function(prec, recall){
  f1score = (2*prec*recall)/(prec+recall)
  print(f1score)
}

#loading in files
result <- read_csv("result.csv")
# change file name to non_removed2 for modified and non_removed for coehlo
non_removed <- read_csv("non_removed.txt", col_names = FALSE)

# removing un-needed files based on correlative analysis
non_removed <- non_removed$X1
non_removed <- append(non_removed,c("Status", "Repository"))
cresult <- result[, which(names(result) %in% non_removed)]

# seperating the status label and repositroy label
status_label <- cresult$Status
repository_label <- cresult$Repository

# removing the labels to do classification
unlabelresult <- cresult[ ,-which(names(cresult) %in% c("Status", "Repository"))]

# cleaning up results with levels of active/inactive
cresult$Status <- lapply(cresult$Status, str_replace, "Archived", "Inactive")
cresult$Status <- lapply(cresult$Status, str_replace, "FSE", "Inactive")

# replicate the following 100 times this is the random forest 
averagemetric <- replicate(100,{
  foldsind <- createFolds(cresult$Status, k=5)
  testind <- foldsind$Fold5

  testData <- cresult[testind, ]
  trainData <- cresult[-testind, ]

  xtest <- testData[ ,-which(names(testData) %in% c("Status","Repository"))]
  xtrain <- trainData[, -which(names(testData) %in% c("Status","Repository"))]

  ytest <- testData$Status
  ytest <- t(as.data.frame(ytest))
  ytest <- factor(ytest)

  ytrain <- trainData$Status
  ytrain <- t(as.data.frame(ytrain))
  ytrain <- factor(ytrain)

  gitRF <- randomForest(x= xtrain, y=ytrain, xtest = xtest, ytest = ytest)
  gitRF$test$predicted

  # Accuracy
  rf.acc <- mean(gitRF$test$predicted == ytest)
  # create an actual and predicted set
  actual <- as.numeric(ytest)
  pred <- gitRF$test$predicted
  # recall
  rf.recall <- recall(ytest, pred)
  # precision
  rf.prec <- precision(ytest, pred)
  # f1score
  rf.f1 <- f1score(precision(ytest, pred), recall(ytest,pred))
  # Cohen Kappa
  cm <- caret::confusionMatrix(pred, ytest)
  rf.kappa <- cm$overall['Kappa'][1]
  # AUROC
  actualnum <- as.numeric(ytest) -1
  prednum <- as.numeric(pred) -1
  rf.auc <- InformationValue::AUROC(actualnum, prednum)
  eval.table <- c(rf.acc, rf.recall,rf.prec,rf.f1,rf.kappa,rf.auc)
})

#randomForest Average metrics after 100 trials
averagelabel = c("Accuracy", "Recall", "Precision", "FScore", "Kappa", "AUC")
row.names(averagemetric) <- averagelabel
avg.df <- lapply(as.data.frame(t(averagemetric)), mean)


#negative control( always saying the repository is inactive)
averagemetricn <- replicate(100,{
  foldsind <- createFolds(cresult$Status, k=5)
  testind <- foldsind$Fold5
  
  testData <- cresult[testind, ]
  
  pred <- factor(sample(c("Active", "Inactive"),size = length(testData$Status), replace = TRUE, prob = c(0,1)), levels = c("Active","Inactive"))

  ytest <- testData$Status
  ytest <- t(as.data.frame(ytest))
  ytest <- factor(ytest)

  # Accuracy
  rf.acc <- mean(pred == ytest)
  # create an actual and predicted set
  actual <- as.numeric(ytest)
  # recall
  rf.recall <- recall(ytest, pred)
  # precision
  rf.prec <- precision(ytest, pred)
  # f1score
  rf.f1 <- f1score(precision(ytest, pred), recall(ytest,pred))
  # Cohen Kappa
  cm <- caret::confusionMatrix(pred, ytest)
  rf.kappa <- cm$overall['Kappa'][1]
  # AUROC
  actualnum <- as.numeric(ytest) -1
  prednum <- as.numeric(pred) -1
  rf.auc <- InformationValue::AUROC(actualnum, prednum)
  eval.table <- c(rf.acc, rf.recall,rf.prec,rf.f1,rf.kappa,rf.auc)
})


row.names(averagemetricn) <- averagelabel
avg.dfn <- lapply(as.data.frame(t(averagemetricn)), mean)


#random predic
averagemetricr <- replicate(100,{
  foldsind <- createFolds(cresult$Status, k=5)
  testind <- foldsind$Fold5
  
  testData <- cresult[testind, ]
  
  pred <- factor(sample(c("Active", "Inactive"),size = length(testData$Status), replace = TRUE, prob = c(.5,.5)))
  ytest <- testData$Status
  ytest <- t(as.data.frame(ytest))
  ytest <- factor(ytest)
  
  # Accuracy
  rf.acc <- mean(pred == ytest)
  # create an actual and predicted set
  actual <- as.numeric(ytest)
  # recall
  rf.recall <- recall(ytest, pred)
  # precision
  rf.prec <- precision(ytest, pred)
  # f1score
  rf.f1 <- f1score(precision(ytest, pred), recall(ytest,pred))
  # Cohen Kappa
  cm <- caret::confusionMatrix(pred, ytest)
  rf.kappa <- cm$overall['Kappa'][1]
  # AUROC
  actualnum <- as.numeric(ytest) -1
  prednum <- as.numeric(pred) -1
  rf.auc <- InformationValue::AUROC(actualnum, prednum)
  eval.table <- c(rf.acc, rf.recall,rf.prec,rf.f1,rf.kappa,rf.auc)
  
})

row.names(averagemetricr) <- averagelabel
avg.dfr <- lapply(as.data.frame(t(averagemetricr)), mean)

avg.df
avg.dfn
avg.dfr
avg.dff <- rbind(as.data.frame(avg.df), as.data.frame(avg.dfn), as.data.frame(avg.dfr))

grid.table(avg.dff)
barplot(avg.dff$Accuracy, col = "Red", ylab = "Accuracy", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "Accuracy of Random Forest Model Compared to Controls", ylim = c(0,1))
barplot(avg.dff$Recall, col = "Blue", ylab = "Recall", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "Recall of Random Forest Model Compared to Controls", ylim = c(0,1))
barplot(avg.dff$Precision, col = "Yellow", ylab = "Precision", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "Precision of Random Forest Model Compared to Controls", ylim = c(0,1))
barplot(avg.dff$FScore, col = "Red", ylab = "F1 Score", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "F1 Score of Random Forest Model Compared to Controls", ylim = c(0,1))
barplot(avg.dff$Kappa, col = "Blue", ylab = "Cohen Kappa", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "Kappa of Random Forest Model Compared to Controls", ylim = c(0,1))
barplot(avg.dff$AUC, col = "Yellow", ylab = "AUROC", names.arg = c("RandomForest", "Negative", "Random"), xlab = "Model", main = "AUROC of Random Forest Model Compared to Controls", ylim = c(0,1))
