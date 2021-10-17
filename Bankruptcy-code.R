#Reading the file
bankruptcy = read.csv("D:/bankruptcy.csv")
dim(bankruptcy)
str(bankruptcy)

#Data Cleaning
##Identifying null value variables
apply(bankruptcy,2,function(x) sum(is.na(x)))
bankruptcy$CUSIP = as.factor(bankruptcy$CUSIP) 
str(bankruptcy)

#Data Splitting
index = sample(nrow(bankruptcy),nrow(bankruptcy)*0.70) 
bankrup_train = bankruptcy[index,] 
bankrup_test = bankruptcy[-index,]

#1) Exploratory Data Analysis
summary(bankrup_train)
#On dependent variable
barplot(table(bankrup_train$DLRSN), main = "Frequency", xlab = "DLRSN", ylab = "Observations")
#On Independent variables
par(mar=c(1,1,1,1))
par(mfrow = c(2,5))
i = 4
for (i in 4:13)
{
  hist((bankrup_train[,i]), main = paste(colnames(bankrup_train[i])), xlab = colnames(bankrup_train[i]), ylab = 'Frequency')}


#2) Model Fitting
library(MASS)
bankrup.glm <- glm(DLRSN ~ R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10, family=binomial, data=bankrup_train) 
summary(bankrup.glm)
AIC(bankrup.glm)
BIC(bankrup.glm)


#3) Improvising the Model
forward_step <- step(bankrup.glm,  direction = "forward")
backward_step <- step(bankrup.glm,  direction = "backward")

#Selection with AIC
#Backward(if nothing is specified)
bankruptcy_back = step(bankrup.glm)
summary(bankruptcy_back)
bankruptcy_back$deviance
AIC(bankruptcy_back)
BIC(bankruptcy_back)

#Selection with BIC
#Backward(if nothing is specified)
##Optimal from AIC and BIC
bankruptcy_back_BIC = step(bankrup.glm, k=log(nrow(bankrup_train)))
summary(bankruptcy_back_BIC)
bankruptcy_back_BIC$deviance
AIC(bankruptcy_back_BIC)
BIC(bankruptcy_back_BIC)

#In-Sample-Prediction
pred_resp <- predict(bankruptcy_back_BIC, type="response")
par(mfrow = c(1,1))
dev.off()
hist(pred_resp)

#Confusion Matrix / mis-classification rate
table(bankrup_train$DLRSN, (pred_resp >0.5)*1, dnn=c("Truth","Predicted"))

#ROC Curve
library(ROCR)
pred = prediction(pred_resp, bankrup_train$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

#Cost Function
cost1 <-function(r, pi, pcut){ 
  weight1 <-35
  weight0 <-1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}

#Asymmetric cost
pcut <-1/(35+1)
cost1(r = bankrup_train$DLRSN, pi = pred_resp, pcut)

#4) Out-of-Sample Prediction
pred_resp_test = predict(bankruptcy_back_BIC, newdata = bankrup_test, type = 'response')
hist(pred_resp_test)
#Confusion Matrix/ mis-classification rate
table(bankrup_test$DLRSN, (pred_resp_test >0.5)*1, dnn=c("Truth","Predicted"))

#ROC Curve
pred = prediction(pred_resp_test, bankrup_test$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

#Cost Function
cost1 <-function(r, pi, pcut){ 
  weight1 <-35
  weight0 <-1
  c1 <- (r==1)&(pi<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(pi>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
}

#Asymmetric cost
pcut <-1/(35+1)
cost1(r = bankrup_test$DLRSN, pi = pred_resp_test, pcut)


#5) Cross validation
costfunc <-function(obs, pred.p)
{
  weight1 <-35
  weight0 <-1
  pcut <-1/(1+weight1/weight0)
  c1 <- (obs==1)&(pred.p < pcut)
  c0 <- (obs==0)&(pred.p >= pcut)
  cost <- mean(weight1*c1 + weight0*c0)
  return(cost)
}

library(boot)
bankruptcy_CV <- glm(DLRSN~ R2 + R3 + R6 + R7 + R8 + R9 + R10, family=binomial, data=bankruptcy)
cv_result <- cv.glm(data=bankruptcy, glmfit=bankruptcy_CV, cost=costfunc, K=10) 
cv_result$delta[2]

#Prediction
pred_resp_full = predict(bankruptcy_CV, newdata = bankruptcy, type = 'response')

#Confusion Matrix/ misclassification rate
table(bankruptcy$DLRSN, (pred_resp_full >0.5)*1, dnn=c("Truth","Predicted"))

#ROC Curve
pred = prediction(pred_resp_full, bankruptcy$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))


#6) Classification trees and neural net

#7)Artificial Neural Network
library(neuralnet)
bankruptcy_neuralnet<- neuralnet(DLRSN ~ R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10, data = bankrup_train, hidden = 6, linear.output = FALSE)
plot(bankruptcy_neuralnet)

#In-Sample-Prediction
bankruptcy.pred1<- neuralnet::compute(bankruptcy_neuralnet, bankrup_train)
head(cbind(bankrup_train$DLRSN, bankruptcy.pred1$net.result), 10)
detach(package:neuralnet)

#ROC
library(ROCR)
pred <- prediction(bankruptcy.pred1$net.result, bankrup_train$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

bankruptcy_pred_neuralnet<- (bankruptcy.pred1$net.result>mean(bankrup_train$DLRSN))*1

#Confusion matrix
table(bankrup_train$DLRSN, bankruptcy_pred_neuralnet, dnn=c("Truth","Predicted"))
#Cost
cost <-function(r, phat){ 
  weight1 <-35
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
} 

#Actual cost
cost(bankrup_train$DLRSN, predict(bankruptcy_neuralnet, bankrup_train, type="prob"))

#Out-of-Sample
bankruptcy.pred1<- neuralnet::compute(bankruptcy_neuralnet, bankrup_test)
head(cbind(bankrup_test$DLRSN, bankruptcy.pred1$net.result), 10)
detach(package:neuralnet)

#ROC
library(ROCR)
pred <- prediction(bankruptcy.pred1$net.result, bankrup_test$DLRSN)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)
#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

bankruptcy_pred_neuralnet<- (bankruptcy.pred1$net.result>mean(bankrup_test$DLRSN))*1

#Confusion matrix
table(bankrup_test$DLRSN, bankruptcy_pred_neuralnet, dnn=c("Truth","Predicted"))
#Cost
cost <-function(r, phat){ 
  weight1 <-35
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
} 

#Actual cost
cost(bankrup_test$DLRSN, predict(bankruptcy_neuralnet, bankrup_test, type="prob"))


##Classification Trees
library(rpart)
library(rpart.plot)
bankruptcy_trees<- rpart(formula = DLRSN~ R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10,data=bankrup_train, method = 'class', parms = list(loss = matrix(c(0,5,1,0), nrow=2)))
#printing and plotting
print(bankruptcy_trees)
prp(bankruptcy_trees, extra = 1)

#Pruning
bankruptcy_largetree = rpart(formula = DLRSN~ R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10, data = bankrup_train, cp = 0.001)
#printing and plotting
prp(bankruptcy_largetree)
plotcp(bankruptcy_largetree)
printcp(bankruptcy_largetree)

treepruned = prune(bankruptcy_largetree, cp = 0.017)
prp(treepruned)

#Optimal tree model
bankruptcy_final = rpart(formula = DLRSN~ R1 + R2 + R3 + R4 + R5 + R6 + R7 + R8 + R9 + R10, data = bankrup_train, cp = 0.017, method = 'class', parms = list(loss = matrix(c(0,5,1,0), nrow=2)))

#In-sample-prediction
bankruptcy_insample<- predict(bankruptcy_final, bankrup_train, type="class")
#Confusion Matrix
table(bankrup_train$DLRSN, bankruptcy_insample, dnn=c("Truth", "Predicted"))
#Cost
cost <-function(r, phat){ 
  weight1 <-35
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
} 

#Actual cost
cost(bankrup_train$DLRSN, predict(bankruptcy_final, bankrup_train, type="prob"))


#Probability of getting 1
bankrup_train_prob_rpart = predict(bankruptcy_final, bankrup_train, type="prob")
#ROC Curve
library(ROCR)
pred = prediction(bankrup_train_prob_rpart[,2], bankrup_train$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))

#Out-of-sample-prediction
bankruptcy_outsample<- predict(bankruptcy_final, bankrup_test, type="class")
#Confusion Matrix
table(bankrup_test$DLRSN, bankruptcy_outsample, dnn=c("Truth", "Predicted"))
#Cost
cost <-function(r, phat){ 
  weight1 <-35
  weight0 <-1
  pcut <- weight0/(weight1+weight0) 
  c1 <- (r==1)&(phat<pcut) #logical vector - true if actual 1 but predict 0
  c0 <-(r==0)&(phat>pcut) #logical vector - true if actual 0 but predict 1
  return(mean(weight1*c1+weight0*c0)) 
} 

#Actual cost
cost(bankrup_test$DLRSN, predict(bankruptcy_final, bankrup_test, type="prob"))


#Probability of getting 1
bankrup_test_prob_rpart = predict(bankruptcy_final, bankrup_test, type="prob")
#ROC Curve
library(ROCR)
pred = prediction(bankrup_test_prob_rpart[,2], bankrup_test$DLRSN)
perf = performance(pred, "tpr", "fpr")
plot(perf, colorize = TRUE)

#AUC
unlist(slot(performance(pred,"auc"),"y.values"))




