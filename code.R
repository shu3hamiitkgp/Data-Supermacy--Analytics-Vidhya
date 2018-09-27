#data Supermacy

data<-read.csv("train.csv",na.strings = c("",NA))
data.test<-read.csv("test.csv",na.strings = c("",NA))

data<-data[-1]
data.test<-data.test[-1]
library(mlr)
summarizeColumns(data)
summarizeColumns(data.test)


prop.table(table(data$target))
summary(data$gender)
plot(data$relevent_experience,data$target)


library(dummies)

# data <- cbind(data, dummy(data$program_type, sep = "_",))
# data
library(caret)
dummy <- dummyVars(~ ., data = data.imp1, fullRank = TRUE)
dummy.data<-predict(dummy,data.imp1)
dummy.data<-as.data.frame(dummy.data)
# dummy.data$pass_ratio<-data1$pass_ratio
dummy.test<-dummyVars(~ ., data = data.test, fullRank = TRUE)#data.test
dummy.data.test<-predict(dummy.test,data.test)
dummy.data.test<-as.data.frame(dummy.data.test)
# dummy.data$pass_ratio<-data1$pass_ratio
###########------extra xgboost (dummy)-----------############
library(xgboost)
train_index <- createDataPartition(data.imp1$target,p=0.7,list=FALSE)
# Full data set
data_variables <- as.matrix(dummy.data[,-177])
data_label <- dummy.data[,"targetY"]
data_matrix <- xgb.DMatrix(data = as.matrix(dummy.data), label = data_label)
# split train data and make xgb.DMatrix
train_data   <- data_variables[train_index,]
train_label  <- data_label[train_index]
train_matrix <- xgb.DMatrix(data = train_data, label = train_label)
# split test data and make xgb.DMatrix
test_data  <- data_variables[-train_index,]
test_label <- data_label[-train_index]
test_matrix <- xgb.DMatrix(data = test_data, label = test_label)

xgb_params <- list("objective" = "binary:logistic",
                   "eval_metric" = "auc","eta"=0.01,"colsample"=0.3,"subsample"=0.8)

nround    <- 1000 # number of XGBoost rounds
cv.nfold  <- 10
# Fit cv.nfold * cv.nround XGB models and save OOF predictions
cv_model <- xgb.cv(params = xgb_params,
                   data = train_matrix, 
                   nrounds = nround,
                   nfold = cv.nfold,
                   verbose = TRUE,
                   prediction = TRUE)
bst_model <- xgb.train(params = xgb_params,
                       data = train_matrix,
                       nrounds = nround)
# Predict hold-out test set
test_pred <- predict(bst_model, newdata = test_matrix)
# test_pred<- ifelse(test_pred>0.5,1,0)
library(ROCR)
predrocr<-ROCR::prediction(test_pred,test_label)#test_pred
perfrocr<-ROCR::performance(predrocr,"tpr","fpr")
plot(perfrocr)
performance(predrocr,"auc")@y.values


#for prediction on the given test set
# test_dataxgb  <- data_variables[-train_index,]
test_label_1 <- sample(c(0,1),size=nrow(dummy.data.test),replace=TRUE)
test_matrix_1 <- xgb.DMatrix(data = as.matrix(dummy.data.test), label = test_label_1)

pred.test<-predict(bst_model,newdata = test_matrix_1)
submission.1<-cbind(ids,pred.test)
submission<-as.data.frame(submission)
names(submission)<-paste(c("id","is_pass"))
write.csv(pred.test,"Solution.csv")
# 0.787 auc
# get the feature real names
names <-  colnames(dummy.data[-177])
# compute feature importance matrix
importance_matrix = xgb.importance(feature_names = names, model = bst_model)
head(importance_matrix)
tail(importance_matrix)
# plot
gp = xgb.plot.importance(importance_matrix[1:20])
print(gp)




#####-----missing value imputation-----#######
install.packages("missForest")
library(missForest)
library(tidyr)
library(dplyr)
data<-data %>% separate(city,c("city","city_number"),"_")
data<-data[-1]


data.imp<-missForest(data,variablewise = TRUE,verbose = TRUE,replace=TRUE)
data.imp1<-data.imp$ximp
write.csv(data.imp1,"Imputed_train.csv")

data.test<-data.test %>% separate(city,c("city","city_number"),"_")
data.test<-data.test[-1]
data.test$city_number<-as.numeric(data.test$city_number)
summary(data.test)
data.test.imp<-missForest(data.test,variablewise = TRUE,verbose = TRUE,replace=TRUE)
data.test.imp1<-data.test.imp$ximp
write.csv(data.test.imp1,"Imputed_test.csv")

  ######-----logistic reg----------#######
data.imp1$target<-ifelse(data.imp1$target==1,"Y","N")
library(caret)
train_index<-createDataPartition(data.imp1$target,p=0.70,list=FALSE)
data.train.lgrg<-data.imp1[train_index,]
data.test.lgrg<-data.imp1[-train_index,]


ctrl <- trainControl(method = "repeatedcv", number = 10,classProbs = TRUE,summaryFunction = twoClassSummary,verboseIter = TRUE,allowParallel = TRUE)
tbmodel <- train(target~., data = data.train.lgrg, method = "glm",trControl = ctrl,metric="ROC")
Tpred <- predict(tbmodel, newdata=data.test.lgrg[-13],type = "prob")
library(pROC)
predrocr<-ROCR::prediction(Tpred$Y ,data.test.lgrg$target)
perfrocr<-ROCR::performance(predrocr,"tpr","fpr")
plot(perfrocr)
performance(predrocr,"auc")@y.values
