################################################################################ 
##############################  Karolina Solarska  #############################
###################################  410858  ###################################
################################################################################ 

library(ROCR)
library(randomForest)
library(xgboost)
library(scorecard)
library(tidyverse)
library(caret)
library(pROC)
library(Information)

Sys.setlocale("LC_TIME", "English")
setwd("...")

# Exercise 1.2
# a) Load the data for the credit risk applications. 
# b) Run necessary preprocessing steps. 
# c) Train logistic regression, random forest and xgboost models. 
# d) Compare the models performance for AUC. Are the ML models better or worse 
# than logistic regression? 
# e) Choose a probability of default threshold level for one of the models and 
# justify your selection.


# a)
data <- read.csv("default_data.csv", stringsAsFactors = T)
str(data)
head(data)

# b)
# First we have to replace "?" with NA

for (col in names(data)) {
  # Check if the column is a factor
  if (is.factor(data[[col]])) {
    # Convert factor to character to replace "?" with NA
    data[[col]] <- as.character(data[[col]])
    data[[col]][data[[col]] == "?"] <- NA
    # Optionally convert back to factor, if desired
    data[[col]] <- factor(data[[col]])
  } 
}

# converting variables to necessary types
data$A1 <- as.integer(ifelse(data$A1 == "a", 0, 1))
data$A2 <- as.character(data$A2)
data$A2 <- as.numeric(data$A2)
data$A4 <- as.integer(match(data$A4, c("l", "u", "y")) - 1)
data$A5 <- as.integer(match(data$A5, c("g", "gg", "p")) - 1)
data$A6 <- as.integer(data$A6)
data$A7 <- as.integer(data$A7)
data$A9 <- as.integer(ifelse(data$A9 == "f", 0, 1))
data$A10 <- as.integer(ifelse(data$A10 == "f", 0, 1))
data$A12 <- as.integer(ifelse(data$A12 == "f", 0, 1))
data$A13 <- as.integer(match(data$A13, c("g", "p", "s")) - 1)
data$A14 <- as.character(data$A14)
data$A14 <- as.numeric(data$A14)
data$default  <- as.factor(data$default)

# We have to also deal with missings
sum(is.na(data))
data <- data[complete.cases(data),]

# data division into training and testing sample
set.seed(123456)
A = sort(sample(nrow(data), nrow(data) * .8))
train <- data[A, ]
test <- data[-A, ]

# c)

model_1 <- default ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + 
  A12 + A13 + A14 + A15

# LOGISTIC REGRESSION

model_lr <- glm(default ~ A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 + A10 + A11 + 
                  A12 + A13 + A14 + A15, data = train, family = binomial())
summary(model_lr)

# Making predictions with the logistic regression model on the training data
predict_lr <- predict(model_lr, type = 'response', newdata = train)

# confusion matrix 
table(train$default, predict_lr > 0.4)

# ROC Curve 
ROCpred_lr <- prediction(predict_lr, train$default)
ROCperf_lr <- performance(ROCpred_lr, 'tpr', 'fpr')
plot(ROCperf_lr)

# AUC value
auc_lr <- performance(ROCpred_lr, measure = "auc")
auc_lr <- auc_lr@y.values[[1]]
auc_lr

# the same with testing sample
predict_LR <- predict(model_lr, type = 'response', newdata = test)

# confusion matrix 
table(test$default, predict_LR > 0.4)

# ROC Curve 
ROCpred_lr_t <- prediction(predict_LR, test$default)
ROCperf_lr_t <- performance(ROCpred_lr_t, 'tpr', 'fpr')
plot(ROCperf_lr_t)

# AUC value
auc_lr <- performance(ROCpred_lr_t, measure = "auc")
auc_lr <- auc_lr@y.values[[1]]
auc_lr

# RANDOM FOREST

# estimating the model
model_rf <- randomForest(model_1, data = train, ntree = 100)
print(model_rf)
plot(model_rf)

# Predictions

# Train
predict_rf <- predict(model_rf, train, type = "prob")[, 1]
ROC_train_rf  <- roc(as.numeric(train$default == 1), predict_rf)

# Test
predict_RF  <- predict(model_rf, test, type = "prob")[, 1]
ROC_test_rf   <- roc(as.numeric(test$default == 1), predict_RF)

# AUC value
auc_rf <- auc(ROC_test_rf)
cat("AUC:", auc_rf, "\n")

# ROC Curve 
list(
  ROC_train_rf   = ROC_train_rf,
  ROC_test_rf    = ROC_test_rf
  
) %>%
  pROC::ggroc(alpha = 0.5, linetype = 1, size = 1) + 
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), 
               color = "grey", 
               linetype = "dashed") +
  labs(title = paste0("Gini TEST: ",
                      "rf = ", 
                      round(100 * (2 * auc(ROC_test_rf) - 1), 1), "%"),
       subtitle =  paste0("Gini TRAIN: ",
                          "rf = ", 
                          round(100 * (2 * auc(ROC_train_rf) - 1), 1), "% ")) +
  theme_bw() + coord_fixed() +
  scale_color_brewer(palette = "Paired")

# XGBOOST 

# Prepare data for xgboost
dtrain <- xgb.DMatrix(data = as.matrix(train[,-ncol(train)]), label = as.numeric(train$default)-1)
dtest <- xgb.DMatrix(data = as.matrix(test[,-ncol(test)]), label = as.numeric(test$default)-1)

# Set parameters
params <- list(
  booster = "gbtree",
  objective = "binary:logistic",
  eta = 0.3,
  gamma = 0,
  max_depth = 6,
  min_child_weight = 1,
  subsample = 1,
  colsample_bytree = 1
)

# Train the model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, verbose = 0)

# Predictions
# On training data
predict_xgb_train <- predict(xgb_model, dtrain)
ROC_train_xgb <- roc(as.numeric(train$default == 1), predict_xgb_train)

# On testing data
predict_xgb_test <- predict(xgb_model, dtest)
ROC_test_xgb <- roc(as.numeric(test$default == 1), predict_xgb_test)

# AUC value
auc_xgb_train <- auc(ROC_train_xgb)
auc_xgb_test <- auc(ROC_test_xgb)
cat("AUC Train:", auc_xgb_train, "\n")
cat("AUC Test:", auc_xgb_test, "\n")

# ROC Curve for XGBoost
roc_curve_xgb <- ggroc(list(Train = ROC_train_xgb, Test = ROC_test_xgb), alpha = 0.5, size = 1) +
  geom_segment(aes(x = 1, xend = 0, y = 0, yend = 1), color = "grey", linetype = "dashed") +
  labs(title = "ROC Curve for XGBoost Model",
       subtitle = paste0("AUC Train: ", round(auc_xgb_train, 2), 
                         ", AUC Test: ", round(auc_xgb_test, 2))) +
  theme_bw() + coord_fixed()

print(roc_curve_xgb)

# d) # AUC

cat("AUC Linear Regression (Test Sample):", auc_lr, "\n", 
    "AUC Random Forest (Test Sample)", auc_rf, "\n",
    "AUC XGBOOST (Test Sample)", auc_xgb_test, "\n")

# Both the Random Forest and XGBoost models perform better than the Logistic 
# Regression model, as indicated by their higher AUC values on the test data so
# the machine learning models are able to better distinguish between the positive 
# and negative classes compared to the simpler logistic regression model.

# e) Choosing a probability of default threshold level 

# Generate a sequence of thresholds
thresholds <- seq(0.01, 0.99, by = 0.01)

test$default <- factor(test$default, levels = c("0", "1"))

# Initialize an empty dataframe to store precision, recall, and F1 scores
results <- data.frame(Threshold = numeric(), F1 = numeric(), Precision = numeric(), Recall = numeric())

for (threshold in thresholds) {
  predicted <- ifelse(predict_RF > threshold, 1, 0)
  # Ensure predicted has the same factor levels as test$default
  predicted <- factor(predicted, levels = c("0", "1"))
  
  # Now, use caret's confusionMatrix
  cm <- confusionMatrix(predicted, test$default)
  
  # Extract metrics
  precision <- cm$byClass['Pos Pred Value']
  recall <- cm$byClass['Sensitivity']  # Recall is the same as Sensitivity
  f1 <- ifelse(is.nan(precision) | is.nan(recall), NA, 2 * ((precision * recall) / (precision + recall)))
  
  # Append to results dataframe
  results <- rbind(results, data.frame(Threshold = threshold, F1 = f1, Precision = precision, Recall = recall))
}

# Remove rows with NA values in F1 scores
results <- na.omit(results)

# Find the threshold with the maximum F1 score
max_f1 <- results[which.max(results$F1),]

print(max_f1)

# The output indicates that the threshold of 0.99 maximizes the F1 score with a 
# value of approximately 0.62 for the Random Forest model predictions on your 
# test dataset. This result suggests a precision of about 0.45 and a recall of 1.

# Alternatively ROC method, based on Youden's Index can be used.
best_threshold <- coords(ROC_test_rf, "best", maximize = TRUE)$threshold
best_threshold

# Plot the ROC curve with the best threshold
plot(ROC_test_rf, print.auc = TRUE, auc.polygon = TRUE, grid = TRUE)
abline(h = 1, v = 1, col = "orange", lty = 2)
points(best_threshold, coords(ROC_test_rf, "best", maximize = "f1")$f1, col = "red", pch = 19)
text(0.5, 0.5, paste("Threshold =", round(best_threshold, 2)), col = "red", pos = 3)

# Chosen threshold: 0.475