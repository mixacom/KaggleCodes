# load the libraries
library(readr) 
library(xgboost)
library(Matrix) 
library(caret) 

# load the dataset and use basic functions because another function gives a lot of errors   
train <- read.csv("/home/mikhail/Desktop/GitProjects/KaggleCodes/business/Santander Customer Satisfaction/data/train.csv", 
                  stringsAsFactors = FALSE) 
test <- read.csv("/home/mikhail/Desktop/GitProjects/KaggleCodes/business/Santander Customer Satisfaction/data/test.csv", 
                 stringsAsFactors = FALSE) 

set.seed(1234)

##### Removing IDs
train$ID <- NULL
test.id <- test$ID
test$ID <- NULL

##### Extracting TARGET
train.y <- train$TARGET
train$TARGET <- NULL

##### 0 count per line
count0 <- function(x) {
    return( sum(x == 0) )
}
train$n0 <- apply(train, 1, FUN=count0)
test$n0 <- apply(test, 1, FUN=count0)

##### Removing constant features
cat("\n## Removing the constants features.\n")
for (f in names(train)) {
    if (length(unique(train[[f]])) == 1) {
        cat(f, "is constant in train. We delete it.\n")
        train[[f]] <- NULL
        test[[f]] <- NULL
    }
} 

# remove columns with a zero std from experiment 
stdColumns <- apply(train, 2, sd) 
train <- train[, stdColumns != 0] 
test <- test[ , stdColumns != 0]  

##### Removing identical features
features_pair <- combn(names(train), 2, simplify = F)
toRemove <- c()
for(pair in features_pair) {
    f1 <- pair[1]
    f2 <- pair[2]
    
    if (!(f1 %in% toRemove) & !(f2 %in% toRemove)) {
        if (all(train[[f1]] == train[[f2]])) {
            cat(f1, "and", f2, "are equals.\n")
            toRemove <- c(toRemove, f2)
        }
    }
}
feature.names <- setdiff(names(train), toRemove)

train <- train[, feature.names]
test <- test[, feature.names]

# remove some values 
train$var3[train$var3 < 0] <- 2L 

# identify columns that are a linear combination of other features 
lin.comb <- findLinearCombos(train) 
train <- train[, -lin.comb$remove] 
test <- test[, -lin.comb$remove] 

# for (f in feature.names) {
#     if (class(train[[f]]) == "numeric" & class(test[[f]]) == "numeric") {
#         train[[f]] <- train[[f]] / max(train[[f]]) 
#         test[[f]] <- test[[f]] / max(test[[f]]) 
#     }
# } 

# nfeatures <- ncol(train) 
# for (i in 1:nfeatures) {
#     train[, i] <- log(train[, i] + 1) 
#     test[, i] <- log(test[, i] + 1) 
# }
# train[is.na(train)] <- -999 
# test[is.na(test)] <- 999  

train$TARGET <- train.y 

# trainMain <- train[1:68418, ]
# trainSecond <- train[68419:76020, ] 

# sb <- as.character(train.y$tvn) 
# sb[sb == "hc"] <- "0" 
# sb[sb == "uc"] <- "1" 
# sb <- as.numeric(sb) 
# train$TARGET <- sb 

train <- sparse.model.matrix(TARGET ~ ., data = train) 

dtrain <- xgb.DMatrix(data=train, label=train.y) 
watchlist <- list(train=dtrain)

test$TARGET <- -1
test <- sparse.model.matrix(TARGET ~ ., data = test)

# target.ds <- trainSecond$TARGET 
# 
# trainSecond$TARGET <- -1 
# dstrain <- sparse.model.matrix(TARGET ~ ., data = trainSecond) 
# dstrain <- xgb.DMatrix(data = dstrain, label=train.y[68419:76020])  

# parameter eta should be equal to 2 / nround 
# subsample equal to 0.75 
# colsample equal to 0.8 

param <- list(  objective           = "binary:logistic", 
                booster             = "gbtree",
                eval_metric         = "auc",
                eta                 = 0.02, 
                max_depth           = 5,
                subsample           = 0.70,
                colsample_bytree    = 0.70,
                nthread             = 4 
) 

cvres <- xgb.cv(params = param, data = dtrain, nrounds = 500, nfold = 2L, metrics = "auc", 
                stratified = TRUE, verbose = TRUE, prediction = FALSE, maximize = FALSE) 

best.iter <- which.max(cvres$test.auc.mean) 

xgb_grid_1 = expand.grid(
    nrounds = c(200, 500), 
    eta = c(0.3, 0.6),
    max_depth = 3,  
    gamma = 0,  
    min_child_weight = 1, 
    colsample_bytree = c(0.2, 0.9) 
) 

# pack the training control parameters
xgb_trcontrol_1 = trainControl(
    method = "cv",
    number = 3,
    verboseIter = TRUE,
    returnData = FALSE,
    returnResamp = "all",                                                        # save losses across all models
    classProbs = TRUE,                                                           # set to TRUE for AUC to be computed
    summaryFunction = twoClassSummary,
    allowParallel = TRUE,
    selectionFunction = "best" 
) 

# names(train) <- make.names(names(train)) 
# train.y <- data.frame(train.y) 
# names(train.y) <- make.names("tvn") 
# train.y$tvn[train.y$tvn == 0] <- "hc" 
# train.y$tvn[train.y$tvn == 1] <- "uc" 
# train.y$tvn <- as.factor(train.y$tvn) 
# yt <- train.y$tvn 

# train the model for each parameter combination in the grid, 
#   using CV to evaluate
xgb_train_1 = train(
    x = as.matrix(train), 
    y = yt,  
    trControl = xgb_trcontrol_1,
    tuneGrid = xgb_grid_1,
    method = "xgbTree",
    maximize = TRUE 
) 

preds <- rep(0, nrow(test)) 

for (z in 1:10) {
    clf <- xgb.train(   params              = param, 
                        data                = dtrain, 
                        nrounds             = 500, 
                        verbose             = 1,  
                        watchlist           = watchlist,
                        maximize            = TRUE 
    )    
    pred <- predict(clf, test)
    preds <- preds + pred
}
preds <- preds / 10.0 

# compResults <- predict(clf, dstrain) 
# cmp <- compResults == target.ds 

submission <- data.frame(ID=test.id, TARGET=preds) 
cat("saving the submission file\n")
write.csv(submission, "../KaggleCodes/submission.csv", row.names = F)  


