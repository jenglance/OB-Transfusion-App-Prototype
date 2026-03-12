library(caret)
library(xgboost)
library(pROC)
library(ggplot2)

# ---- Step 1: Load data ----
data <- read.csv("C:/JenApps/REV_Train.csv", stringsAsFactors = TRUE, header = TRUE)

# ---- Step 2: Prepare data ----
# Convert Cesarean indications to factor
data$Cesarean.Indications <- factor(data$Cesarean.Indications)

# Create mapping of factor levels (optional)
ind_levels <- levels(data$Cesarean.Indications)
mapping_df <- data.frame(LevelCode = seq_along(ind_levels), LevelName = ind_levels)
print(mapping_df)

# Create factor levels list for Shiny
factor_levels <- list(Cesarean.Indications = ind_levels)

# Create binary target variable
data$churn <- factor(ifelse(data$Units == 0, "no", "yes"), levels = c("no", "yes"))

# Remove ID and Units to prevent leakage
data_model <- data[, !(names(data) %in% c("Units"))]

# ---- Step 3: Train/test split ----
set.seed(123)
trainIndex <- createDataPartition(data_model$churn, p = 0.8, list = FALSE)
trainData <- data_model[trainIndex, ]
testData  <- data_model[-trainIndex, ]

# Remove single-level factors
single_level_factors <- sapply(trainData, function(col) is.factor(col) && length(unique(col)) < 2)
if (any(single_level_factors)) trainData <- trainData[, !single_level_factors]
testData <- testData[, names(trainData)]

# ---- Step 4: Train XGBoost model ----
ctrl <- trainControl(
  method = "cv",
  number = 10,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  verboseIter = TRUE
)

xgb_grid <- expand.grid(
  nrounds = c(50, 100),
  max_depth = c(3, 6),
  eta = 0.1,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.8
)

set.seed(123)
xgb_model <- train(
  churn ~ .,
  data = trainData,
  method = "xgbTree",
  metric = "ROC",
  trControl = ctrl,
  tuneGrid = xgb_grid
)

# ---- Step 5: Evaluate model ----
# Predicted probabilities
pred_probs <- predict(xgb_model, newdata = testData, type = "prob")[, "yes"]

# ROC object
roc_obj <- roc(
  response = testData$churn,
  predictor = pred_probs,
  levels = c("no", "yes"),
  direction = "<"
)

# Optimal threshold (Youden's J)
opt_coords <- coords(
  roc_obj,
  "best",
  ret = c("threshold", "sensitivity", "specificity", "precision", "recall", "accuracy"),
  best.method = "youden"
)
opt_thresh <- as.numeric(opt_coords["threshold"])

# Confusion matrix and metrics
pred_class <- factor(ifelse(pred_probs >= opt_thresh, "yes", "no"), levels = c("no", "yes"))
cm <- confusionMatrix(pred_class, testData$churn, positive = "yes")

metrics <- data.frame(
  Metric = c("AUC", "Accuracy", "Sensitivity", "Specificity", "Precision", "F1"),
  Value = round(c(
    auc(roc_obj),
    cm$overall["Accuracy"],
    cm$byClass["Sensitivity"],
    cm$byClass["Specificity"],
    cm$byClass["Pos Pred Value"],
    2 * (cm$byClass["Pos Pred Value"] * cm$byClass["Sensitivity"]) /
      (cm$byClass["Pos Pred Value"] + cm$byClass["Sensitivity"])
  ), 4)
)

# ---- Step 6: Feature importance (optional) ----
xgb_final <- xgb_model$finalModel
importance_df <- xgb.importance(model = xgb_final)[1:10, ]
ggplot(importance_df, aes(x = reorder(Feature, Gain), y = Gain)) +
  geom_col(fill = "darkblue") +
  coord_flip() +
  labs(title = "Top 10 Important Features", x = "Feature", y = "Gain") +
  theme_minimal()

# ---- Step 7: Save model bundle for Shiny ----
model_bundle <- list(
  model = xgb_model,
  roc = roc_obj,
  best_threshold = opt_thresh,
  threshold_metrics = opt_coords,
  metrics = metrics,
  factor_levels = factor_levels
)

saveRDS(model_bundle, "C:/JenApps/xgb_model_bundle.rds")
cat("✅ Model bundle saved as 'xgb_model_bundle.rds'\n")
cat("Included objects:\n")
print(names(model_bundle))