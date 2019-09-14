#load libraries 
library(randomForest)
library(xgboost)
library(Matrix)
library(nnet)
library(data.table)
library(caret)
library(corrplot)
library(foreach)
library(doSNOW)
library(dplyr)


#load data 

mydata<- read.csv("C://Users/s137r4//Desktop//Analysisfile//ACC17.csv")

mydata <- mydata[, -c(1:2, 3:6)] #delete the indices and location vars #145941 obs of 64 vars


#Filter vehicles 

mydata<-mydata %>%
  filter(Vehicle_Type %in% (8:9)) #96890 obs

#delete irrelevant features
mydata<-mydata[, -c(4:5,7:9,43,54:55,50,27:28,62)]

#change categorical varaibles to factors
colz <- c(2,4,7:11,13:34,37:50)
mydata[, colz]<-lapply(mydata[,colz], factor)

#split the data into training and test data
set.seed(1234)
ind<- sample(2, nrow(mydata), replace =T, prob =c(0.8,0.2))

train <- mydata[ind==1, ]
test <- mydata[ind==2, ]

##XGBOOST estimation 

#One hot coding and 
#train matrix 
predictors<-train[, -c(2)] #all predictors excluding accident severity
label = as.numeric(train$Accident_Severity)-1
dataxgb<- as.matrix(predictors)
mode(dataxgb) <- 'double' # to numeric i.e double precision 

#test matrix
test.predictors <- test[, -c(2)]
test_data<- as.matrix(test.predictors)
mode(test_data) <- 'double'

#tuning xgboost parameters
#step 1: max_depth, min_child_weight, and learning rate

nrounds=100
tune_grid <- expand.grid(
  nrounds = seq(from =5, to=nrounds, by=10),
  max_depth = c(4,6,7),
  eta= c(0.1,0.3),
  gamma=0,
  colsample_bytree = 0.8,
  min_child_weight = c(1,2,3,4), 
  subsample =1
)

xgb_tune <- caret::train(
  x=dataxgb,
  y=label,
  trcontrol=trainControl(method = "cv", number = 3, verboseIter = T), #crossvalidation
  tuneGrid = tune_grid,
  method ="xgbTree",
  verbose=T
 
)
#obtain the best tune 

xgb_tune$bestTune


#run the model using XGBOOST package to obtain the results 

#Set parameters for the model, since our response variable is multilclass, we will use multisoft as the objective and the number os classes are 3
param = list("objective" = "multi:softmax", # multi class classification
                   "num_class"= 3 ,  		# Number of classes in the dependent variable.
                   "eval_metric" = "mlogloss",  	 # evaluation metric 
                   "nthread" = 4,   			 # number of threads to be used 
                   "max_depth" = 6,    		 # maximum depth of tree 
                   "eta" = 0.3,    			 # step size shrinkage #default0.3
                   "gamma" = 0,    			 # minimum loss reduction #default
                   "subsample" = 0.7,    		 # part of data instances to grow tree. default 1 
                   "colsample_bytree" = 1, 		 # subsample ratio of columns when constructing each tree default -1
                   "min_child_weight" = 2 		 # minimum sum of instance weight needed in a child
)

#perform crossvalidation like in the tuning stage

cv.nround = 200;  # Number of rounds. This can be set to a lower or higher value, if you wish, example: 150 or 250 or 300  
bst.cv = xgb.cv(param=param,
                data = dataxgb,
                label = label,
                nfold = 3,
                nrounds=cv.nround,
                prediction=T)

#find the min mlogloss that occured and on which iteration
names(bst.cv$evaluation_log)
#Find where the minimum logloss occurred
min.loss.idx = which.min(bst.cv$evaluation_log[, test_mlogloss_mean]) 
cat ("Minimum logloss occurred in round : ", min.loss.idx, "\n")
# Minimum logloss
print(bst.cv$evaluation_log[min.loss.idx,])

#Run the model usng the minimum logloss from above.
set.seed(100)
bst = xgboost(
  param=param,
  data =dataxgb,
  label = label,
  nrounds=min.loss.idx)

#Make prediction on the testing data.

testxgb$prediction = predict(bst, test_data) #added a new column with the respective predictors 
testxgb$prediction


#accuracy 

#convert the prediction as factor, as the confusion matrix requires the data to be a factor and theshould have th same levels 
testxgb$prediction<-as.factor(testxgb$prediction)

#convert the referenc variable into numeric and then reduce the variables to start from 0 as xgboost requires the letters to start from 0
testxgb$Accident_Severity<-as.numeric(testxgb$Accident_Severity)-1
testxgb$Accident_Severity<-as.factor(testxgb$Accident_Severity)

#Compute the accuracy of predictions.

confusionMatrix( testxgb$prediction,testxgb$Accident_Severity)

#confusion matrix trainxgb data 

#comupute accuracy for the training data 

trainxgb$prediction=predict(bst, dataxgb)
trainxgb$prediction

#convert the prediction as factor, as the confusion matrix requires the data to be a factor and theshould have th same levels 
trainxgb$prediction<-as.factor(trainxgb$prediction)

#convert the referenc variable into numeric and then reduce the variables to start from 0 as xgboost requires the letters to start from 
trainxgb$Accident_Severity<-as.numeric(trainxgb$Accident_Severity)-1
trainxgb$Accident_Severity<-as.factor(trainxgb$Accident_Severity)



#confusion matrix training matrix - 0.9724

confusionMatrix( trainxgb$prediction,trainxgb$Accident_Severity)

#variable imoortance #gain is the improvement in accuracy brought by a feature to the branches it is on

xgb.importance(model=bst)

importance<- xgb.importance(feature_names =colnames(dataxgb), model = bst)
xgb.plot.importance(importance_matrix = importance)

importance<- xgb.importance(feature_names =colnames(dataxgb), model = bst)
xgb.plot.importance(importance_matrix = importance, top_n = 20)


################## Multinomial logistic regression 
library(e1071)

#Multinomial logistic regression .
mmlog <- caret::train(
  Accident_Severity~.,
  data=train,
  method="multinom",
  trcontrol =trainControl(method = "cv",number = 3), #cross validation 
  linout =T,
  MaxNWts=5244
)

#model output
mmlog
#Training accurracy
preds <- predict(mmlog, train)

#confusion matrix 

cmmultinomial <- table(predict(mmlog), train$Accident_Severity)
print(cmmultinomial)

#misclassification 
1- sum(diag(cmmultinomial))/sum(cmmultinomial)


###testing accuracy
predtest1 <- predict(mmlog, test)

#test confusion matrix
cmmultinomialt <- table(predtest1, test$Accident_Severity)
print(cmmultinomialt)

#testing misclassification
1- sum(diag(cmmultinomialt))/sum(cmmultinomialt)

#######Naive Bayes

#model tuning parameters

x<- subset(train, select = - Accident_Severity)
y<- train$Accident_Severity
searchgrid<- expand.grid(
  usekernel =TRUE,
  fL=0,
  adjust=1) #tuning parameters 


#run the model
nb1<- train(x,y,method="nb",
            trControl = trainControl(method = "cv",  number = 3), tuneGrid = searchgrid
)

#model summary
nb1

#testing accuracy
predict<-predict(nb1, test)


confusionMatrix(predict, test$Accident_Severity)

#train accuracy
predict <- predict(nb1, train)


############################################### Random Forest #####
#grid search
tgrid <- expand.grid(
  .mtry =c(2,3,4),
  .splitrule = "gini",
  .min.node.size =c(10,20)
)


#run the model 

rfcaret <- train(Accident_Severity~., data=train,
                 method="ranger",
                 trControl = trainControl(method= "cv", number = 3, verboseIter = T),
                 tuneGrid = tgrid,
                 num.trees = 100)

#model summary 
rfcaret

#test accuracy
preds <- predict(rfcaret, test)

confusionMatrix(pred, test$Accident_Severity)

#traing accuracy
predstrain <- predict(rfcaret, train)

confusionMatrix(predstrain, test$Accident_Severity)



##variable importance 

rfmp <- varImp(rfcaret, scale = FALSE)
plot(rfmp, top=20)


setwd("C://Users/s137r4//Desktop//Analysisfile//")
knitr::stitch("myrsript.r")

