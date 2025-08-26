# ────────────────────────────────────────────────────────────────────────────────
# 0.   Setup
# ────────────────────────────────────────────────────────────────────────────────

library(tree)
library(e1071)
library(ROCR)
library(randomForest)
library(adabag)
library(rpart)
library(smotefamily)
library(neuralnet)
library(xgboost)

rm(list = ls())

# ────────────────────────────────────────────────────────────────────────────────
# 1.   Loading and Creating My Dataset
# ────────────────────────────────────────────────────────────────────────────────

set.seed(33270961)
WD <- read.csv("WinnData.csv")
WD <- WD[sample(nrow(WD), 5000, replace = FALSE), ]
WD <- WD[, c(sort(sample(1:30, 20, replace = FALSE)), 31)]

# ────────────────────────────────────────────────────────────────────────────────
# 2.   Exploratory Data Analysis
# ────────────────────────────────────────────────────────────────────────────────

# Display structure of dataset
str(WD)

# Count missing values per column
colSums(is.na(WD))

# Examine class distribution
class_counts <- table(WD$Class)
class_props  <- prop.table(class_counts)
print(class_counts)
print(round(class_props, 3))

# Get all predictor variables (exclude target variable "Class")
predictors <- setdiff(names(WD), "Class")

# Summary statistics for predictors only
summary(WD[predictors])

# Calculate standard deviations for all predictors
sapply(WD[predictors], sd)

# Correlation matrices (all variables including Class)
corr_mat_full <- cor(WD, use = "pairwise.complete.obs")
print(round(corr_mat_full, 3))

# ────────────────────────────────────────────────────────────────────────────────
# 3.   Pre-processing
# ────────────────────────────────────────────────────────────────────────────────

# Apply log transformation to A26 (handles zeros with log1p)
WD$A26 <- log1p(WD$A26)

# Remove columns A16 and A19 from dataset
WD <- WD[, ! names(WD) %in% c("A16", "A19")]

# Check structure after transformations
str(WD)

# Update predictor list after column removal
predictors <- setdiff(names(WD), "Class")

# Summary statistics for remaining predictors
summary(WD[predictors])

# Convert Class to factor with meaningful labels
WD$Class <- factor(
  WD$Class,
  levels = c(0, 1),
  labels = c("Other", "Oats")
)

# ────────────────────────────────────────────────────────────────────────────────
# 4.   Train–Test Split
# ────────────────────────────────────────────────────────────────────────────────

set.seed(33270961)

train.row = sample(1:nrow(WD), 0.7*nrow(WD))
WD.train = WD[train.row,]
WD.test = WD[-train.row,]


# ────────────────────────────────────────────────────────────────────────────────
# 5.   Model Training
# ────────────────────────────────────────────────────────────────────────────────

WD.tree  <- tree(Class ~ ., data = WD.train)
WD.nb    <- naiveBayes(Class ~ ., data = WD.train)
WD.bag   <- bagging(Class ~ ., data = WD.train, mfinal = 200)
WD.boost <- boosting(Class ~ ., data = WD.train, mfinal = 200)
WD.rf    <- randomForest(Class ~ ., data = WD.train)

# ────────────────────────────────────────────────────────────────────────────────
# 6.   Evaluation Helper Function
# ────────────────────────────────────────────────────────────────────────────────

evaluate_model <- function(pred, actual) {
  cm <- table(Predicted = factor(pred, levels = c("Other", "Oats")),
              Actual    = factor(actual, levels = c("Other", "Oats")))
  TP <- cm["Oats",  "Oats"]
  TN <- cm["Other","Other"]
  FP <- cm["Oats",  "Other"]
  FN <- cm["Other","Oats"]
  
  accuracy  <- (TP + TN) / sum(cm)
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall    <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1_score  <- ifelse(is.na(precision) | is.na(recall) | (precision + recall) == 0, NA,
                      2 * precision * recall / (precision + recall))
  metrics <- c(
    Accuracy  = round(accuracy,  3),
    Precision = round(precision, 3),
    Recall    = round(recall,    3),
    F1_Score  = round(f1_score,  3)
  )
  list(confusion = cm, metrics = metrics)
}

# ────────────────────────────────────────────────────────────────────────────────
# 7.   Predictions & Metrics
# ────────────────────────────────────────────────────────────────────────────────

# Decision Tree
eval_tree <- evaluate_model(
  predict(WD.tree, WD.test, type = "class"),
  WD.test$Class
)
print(eval_tree$confusion); print(eval_tree$metrics)

# Naive Bayes
cat("\n# Naive Bayes\n")
eval_nb <- evaluate_model(
  predict(WD.nb, WD.test, type = "class"),
  WD.test$Class
)
print(eval_nb$confusion); print(eval_nb$metrics)

# Bagging
eval_bag <- evaluate_model(
  predict.bagging(WD.bag, WD.test)$class,
  WD.test$Class
)
print(eval_bag$confusion); print(eval_bag$metrics)

# Boosting
eval_boost <- evaluate_model(
  predict.boosting(WD.boost, WD.test)$class,
  WD.test$Class
)
print(eval_boost$confusion); print(eval_boost$metrics)

# Random Forest
eval_rf <- evaluate_model(
  predict(WD.rf, WD.test),
  WD.test$Class
)
print(eval_rf$confusion); print(eval_rf$metrics)

# ────────────────────────────────────────────────────────────────────────────────
# 8.   ROC Curves & AUC
# ────────────────────────────────────────────────────────────────────────────────

# Base ROC plot
plot(0, type="n", xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main="ROC Curves of Different Classifiers")
abline(0, 1)

# Decision Tree
tree_scores <- predict(WD.tree, WD.test, type="vector")[, 2]
tree_obj    <- ROCR::prediction(tree_scores, WD.test$Class, label.ordering=c("Other","Oats"))
tree_perf   <- performance(tree_obj, "tpr", "fpr")
plot(tree_perf, add=TRUE, col="orange")
tree_auc    <- performance(tree_obj, "auc")@y.values

# Naive Bayes
nb_scores  <- predict(WD.nb, WD.test, type="raw")[, 2]
nb_obj     <- ROCR::prediction(nb_scores, WD.test$Class, label.ordering=c("Other","Oats"))
nb_perf    <- performance(nb_obj, "tpr", "fpr")
plot(nb_perf, add=TRUE, col="blueviolet")
nb_auc     <- performance(nb_obj, "auc")@y.values

# Bagging
bag_scores <- predict.bagging(WD.bag, WD.test)$prob[, 2]
bag_obj    <- ROCR::prediction(bag_scores, WD.test$Class, label.ordering=c("Other","Oats"))
bag_perf   <- performance(bag_obj, "tpr", "fpr")
plot(bag_perf, add=TRUE, col="blue")
bag_auc    <- performance(bag_obj, "auc")@y.values

# Boosting
boost_scores <- predict.boosting(WD.boost, WD.test)$prob[, 2]
boost_obj    <- ROCR::prediction(boost_scores, WD.test$Class, label.ordering=c("Other","Oats"))
boost_perf   <- performance(boost_obj, "tpr", "fpr")
plot(boost_perf, add=TRUE, col="red")
boost_auc    <- performance(boost_obj, "auc")@y.values

# Random Forest
rf_scores <- predict(WD.rf, WD.test, type="prob")[, 2]
rf_obj    <- ROCR::prediction(rf_scores, WD.test$Class, label.ordering=c("Other","Oats"))
rf_perf   <- performance(rf_obj, "tpr", "fpr")
plot(rf_perf, add=TRUE, col="darkgreen")
rf_auc    <- performance(rf_obj, "auc")@y.values

legend("bottomright",
       legend = c("Decision Tree","Naive Bayes","Bagging","Boosting","Random Forest"),
       col    = c("orange","blueviolet","blue","red","darkgreen"),
       lty    = 1, cex = 0.8)

cat("\n# AUCs:\n",
    "Decision Tree: ", as.numeric(tree_auc), "\n",
    "Naive Bayes  : ", as.numeric(nb_auc), "\n",
    "Bagging      : ", as.numeric(bag_auc), "\n",
    "Boosting     : ", as.numeric(boost_auc), "\n",
    "Random Forest: ", as.numeric(rf_auc), "\n")

# ────────────────────────────────────────────────────────────────────────────────
# 9.   Variable Importance
# ────────────────────────────────────────────────────────────────────────────────

cat("\n# Tree Importance\n");  print(summary(WD.tree));  plot(WD.tree);  text(WD.tree, pretty = 0)
cat("\n# Bagging Importance\n");  print(WD.bag$importance)
cat("\n# Boosting Importance\n");  print(WD.boost$importance)
cat("\n# RF Importance\n");  print(WD.rf$importance)

# ────────────────────────────────────────────────────────────────────────────────
# 10.  New Decision Tree
# ────────────────────────────────────────────────────────────────────────────────

# Train new decision tree with selected features only
WD.new_tree <- tree(Class ~ A26 + A02 + A06, data = WD.train)
print(summary(WD.new_tree))
plot(WD.new_tree); text(WD.new_tree, pretty = 0)

# Evaluate the model
eval_new <- evaluate_model(
  predict(WD.new_tree, WD.test, type="class"),
  WD.test$Class
)
print(eval_new$confusion); print(eval_new$metrics)

# Plot ROC
new_tree_scores <- predict(WD.new_tree, WD.test, type="vector")[, 2]
new_tree_obj    <- ROCR::prediction(new_tree_scores, WD.test$Class, label.ordering=c("Other","Oats"))
new_tree_perf   <- performance(new_tree_obj, "tpr", "fpr")

plot(0, type="n", xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main="ROC Curve - New Decision Tree")
abline(0, 1)
plot(new_tree_perf, add=TRUE, col="orange")

# Compute AUC
new_tree_auc <- performance(new_tree_obj, "auc")@y.values
cat("\n# New Decision Tree AUC\n", as.numeric(new_tree_auc), "\n")

# ────────────────────────────────────────────────────────────────────────────────
# 11.  Best Tree-Based Classifier
# ────────────────────────────────────────────────────────────────────────────────

set.seed(33270961)

# SMOTE to address class imbalance by generating synthetic Oats samples
smote_result <- SMOTE(X = WD.train[, setdiff(names(WD.train), "Class")],
                      target = WD.train$Class, K = 5, dup_size = 5)

# Create new balanced training set from SMOTE results
WD.train_balanced <- data.frame(smote_result$data)
names(WD.train_balanced)[ncol(WD.train_balanced)] <- "Class"
WD.train_balanced$Class <- factor(WD.train_balanced$Class, levels = c("Other", "Oats"))

# Check class distribution
table(WD.train_balanced$Class)

# Split test data into validation and final test sets (50/50)
test_idx <- sample(seq_len(nrow(WD.test)), size = 0.5 * nrow(WD.test))
WD.val   <- WD.test[-test_idx, ]
WD.final_test  <- WD.test[test_idx, ]

# Define hyperparameter grid
mtry_values <- c(2, 3, 4, 5)
ntree_values <- c(500, 750, 1000)
nodesize_values <- c(1, 5, 10)

# Initialize variables
WD.rf_tuned <- NULL
best_model <- NULL
best_f1 <- -Inf

# Grid search for the best hyperparameters
for (mtry in mtry_values) {
  for (ntree in ntree_values) {
    for (nodesize in nodesize_values) {
      # Train the random forest model
      model <- randomForest(
        Class ~ A26 + A02 + A07 + A08 + A14,
        data = WD.train_balanced,
        mtry = mtry,
        ntree = ntree,
        nodesize = nodesize
      )
      pred <- predict(model, WD.val)
      f1 <- evaluate_model(pred, WD.val$Class)$metrics["F1_Score"]
      # Update the best model if F1-score improves
      if (!is.na(f1) && f1 > best_f1) {
        best_f1 <- f1
        WD.rf_tuned <- model  # Save the actual best model
        best_params <- list(mtry = mtry, ntree = ntree, nodesize = nodesize)
      }
    }
  }
}
# Print the best parameters and model
print(best_params)
print(WD.rf_tuned)

# Predict on the test set
rf_tuned_pred <- predict(WD.rf_tuned, WD.final_test)
rf_tuned_scores <- predict(WD.rf_tuned, WD.final_test, type="prob")[, 2]

# Evaluate the tuned model
eval_rf_tuned <- evaluate_model(rf_tuned_pred, WD.final_test$Class)
print(eval_rf_tuned$confusion); print(eval_rf_tuned$metrics)

# Plot ROC
rf_tuned_obj  <- ROCR::prediction(rf_tuned_scores, WD.final_test$Class, label.ordering = c("Other", "Oats"))
rf_tuned_perf <- performance(rf_tuned_obj, "tpr", "fpr")

plot(0, type="n", xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main="ROC Curve - New Random Forest")
abline(0, 1)
plot(rf_tuned_perf, add=TRUE, col="red")

# Compute AUC
rf_tuned_auc <- performance(rf_tuned_obj, "auc")@y.values
cat("\n# Tuned Random Forest AUC\n", as.numeric(rf_tuned_auc), "\n")

# ────────────────────────────────────────────────────────────────────────────────
# 12.  Artificial Neural Network
# ────────────────────────────────────────────────────────────────────────────────

# Make copies for ANN
WD.train_balanced_nn <- WD.train_balanced
WD.test_nn           <- WD.test

# Add numeric outcome
WD.train_balanced_nn$ClassNum <- ifelse(WD.train_balanced_nn$Class == "Oats", 1, 0)
WD.test_nn$ClassNum           <- ifelse(WD.test_nn$Class == "Oats", 1, 0)

# Define specific predictor columns
predictors <- c("A26", "A02", "A06", "A07", "A08", "A14")

# Calculate means and SDs from the balanced training set only
means <- sapply(WD.train_balanced_nn[, predictors], mean)
sds   <- sapply(WD.train_balanced_nn[, predictors], sd)

# Scale predictors in train and test using training set stats
WD.train_balanced_nn[, predictors] <- scale(WD.train_balanced_nn[, predictors], center = means, scale = sds)
WD.test_nn[, predictors] <- scale(WD.test_nn[, predictors], center = means, scale = sds)

# Fit a simple neural network
fmla <- as.formula(paste("ClassNum ~", paste(predictors, collapse = " + ")))
nn_model <- neuralnet(
  fmla,
  data          = WD.train_balanced_nn,
  hidden        = c(5),
  linear.output = FALSE
)

# Predict on the test set
nn_out    <- compute(nn_model, WD.test_nn[, predictors])
nn_scores <- as.vector(nn_out$net.result)
nn_pred   <- factor(ifelse(nn_scores >= 0.5, "Oats", "Other"),
                    levels = c("Other","Oats"))

# Evaluate the ANN
eval_nn <- evaluate_model(nn_pred, WD.test_nn$Class)
print(eval_nn$confusion); print(eval_nn$metrics)

# Plot ROC
nn_obj <- ROCR::prediction(nn_scores, WD.test_nn$ClassNum)
nn_perf <- performance(nn_obj, "tpr", "fpr")

plot(0, type="n", xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main="ROC Curve - Artificial Neural Network")
abline(0, 1)
plot(nn_perf, add=TRUE, col="blue")

# Compute AUC
nn_auc <- performance(nn_obj, "auc")@y.values
cat("\n# ANN AUC\n", as.numeric(nn_auc), "\n")

# ────────────────────────────────────────────────────────────────────────────────
# 13.  XGBoost
# ────────────────────────────────────────────────────────────────────────────────

# Get all predictor variables (exclude target Class)
predictors <- setdiff(names(WD.train), "Class")

# Create XGBoost training matrix with numeric outcome
dtrain <- xgb.DMatrix(
  data  = as.matrix(WD.train[, predictors]),
  label = ifelse(WD.train$Class == "Oats", 1, 0)
)
# Create XGBoost testing matrix with numeric outcome
dtest <- xgb.DMatrix(
  data  = as.matrix(WD.test[, predictors]),
  label = ifelse(WD.test$Class == "Oats", 1, 0)
)

# Five-fold CV to pick number of rounds
params <- list(
  booster            = "gbtree",
  objective          = "binary:logistic",
  eval_metric        = "auc",
  max_depth          = 6,
  eta                = 0.05,
  subsample          = 0.8,
  colsample_bytree   = 0.8,
  scale_pos_weight   = nrow(WD.train[WD.train$Class=="Other",]) / 
    nrow(WD.train[WD.train$Class=="Oats",])
)
cv <- xgb.cv(params, dtrain,
             nrounds = 400,
             nfold = 5,
             early_stopping_rounds = 20,
             verbose = 0)
best_n <- cv$best_iteration

# Train final model
xgb_fit <- xgb.train(params, dtrain, nrounds = best_n)

# Predict on the test set
xgb_prob  <- predict(xgb_fit, dtest)
xgb_class <- factor(ifelse(xgb_prob >= 0.5, "Oats", "Other"),
                    levels = c("Other","Oats"))

# Evaluate the model
eval_xgb <- evaluate_model(xgb_class, WD.test$Class)
print(eval_xgb$confusion); print(eval_xgb$metrics)

# Plot ROC 
xgb_obj <- ROCR::prediction(xgb_prob, WD.test$Class, label.ordering = c("Other", "Oats"))
xgb_perf <- performance(xgb_obj, "tpr", "fpr")

plot(0, type="n", xlim=c(0,1), ylim=c(0,1),
     xlab="False Positive Rate", ylab="True Positive Rate",
     main="ROC Curve - XGBoost")
abline(0, 1)
plot(xgb_perf, add=TRUE, col="green")

# Compute AUC
xgb_auc <- performance(xgb_obj, "auc")@y.values
cat("\n# XGBoost AUC\n", as.numeric(xgb_auc), "\n")

