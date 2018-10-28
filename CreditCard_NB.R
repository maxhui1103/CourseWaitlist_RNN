#install.packages("ggplot2")
library(ggplot2)
#install.packages("dplyr")
library(dplyr)
#install.packages("sm")
library(sm)
#install.packages("caret")
library(caret)
#install.packages("naivebayes")
library(naivebayes)
#install.packages("psych")
library(psych)

credit = read.csv("creditcard.csv")
str(credit)
credit$Class = as.factor(credit$Class)
{ 
  slices = c(sum(credit$Class == 1), sum(credit$Class == 0))
  pct = round(slices/sum(slices)*100, digits = 4)
  lbls = c("Yes", "No")
  lbls = paste(lbls, pct) 
  lbls = paste(lbls, "%", sep = "")
  pie(slices, labels = lbls, col = rainbow(length(lbls)), main = "Yes Vs No")
  remove(lbls, pct, slices)
}

credit$Time_Hr = credit$Time/3600
{
  par(mfcol = c(2, 1))
  hist(credit$Time_Hr[credit$Class == 0], breaks = 48, main = "Genuine", xlab = "Time(Hrs)", xlim = c(0, 50), col = "green")
  hist(credit$Time_Hr[credit$Class == 1], breaks = 48, main = "Fraud", xlab = "Time(Hrs)", xlim = c(0, 50), col = "red")
}
data = subset(credit, select = -Time)

## Case-NB-1 : do not drop anything
# shuffle and split
{
set.seed(1234)
ind = sample(2, nrow(credit), replace = T, prob = c(0.8, 0.2))
train1 = credit[ind == 1, ]
test1 = credit[ind == 2, ]
remove(ind)
}
model1 = naive_bayes(Class ~ ., data = train1, usekernel = T, laplace = 1)

# plot to view distributions of attributes
plot(model1)

p1 = predict(model1, train1)
(tab1 = table(p1, train1$Class))
paste("Recall:", (tab1[2,2]/(tab1[1,2]+tab1[2,2])))
paste("Precision:", (tab1[2,2]/(tab1[2,1]+tab1[2,2])))

p1_test = predict(model1, test1)
(tab1_test = table(p1_test, test1$Class))
paste("Recall:", (tab1_test[2,2]/(tab1_test[1,2]+tab1_test[2,2])))
paste("Precision:", (tab1_test[2,2]/(tab1_test[2,1]+tab1_test[2,2])))

## Case-NB-1 : drop some attributes
# shuffle and split 
credit2 = subset(credit, select = -c(Amount, Time_Hr, V28, V27, V26, V25, V24, V23, V22, V20, V15, V13, V8))
{
set.seed(1234)
ind = sample(2, nrow(credit2), replace = T, prob = c(0.8, 0.2))
train2 = credit2[ind == 1, ]
test2 = credit2[ind == 2, ]
remove(ind)
}
model2 = naive_bayes(Class ~ ., data = train2, usekernel = T, laplace = 1)

p2 = predict(model2, train2)
(tab2 = table(p2, train2$Class))
paste("Recall:", (tab2[2,2]/(tab2[1,2]+tab2[2,2])))
paste("Precision:", (tab2[2,2]/(tab2[2,1]+tab2[2,2])))

p2_test = predict(model2, test2)
(tab2_test = table(p2_test, test2$Class))
paste("Recall:", (tab2_test[2,2]/(tab2_test[1,2]+tab2_test[2,2])))
paste("Precision:", (tab2_test[2,2]/(tab2_test[2,1]+tab2_test[2,2])))
