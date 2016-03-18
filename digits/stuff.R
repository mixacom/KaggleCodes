library(randomForest)
library(readr)
library(caret) 
library(h2o) 

set.seed(0)

numTrain <- 10000
numTrees <- 25

train <- read_csv("/home/mikhail/Desktop/GitProjects/KaggleCodes/digits/input/train.csv") 
test <- read_csv("/home/mikhail/Desktop/GitProjects/KaggleCodes/digits/input/test.csv") 

rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1]

flds <- createFolds(y, k = 10, list = TRUE, returnTrain = FALSE)
names(flds)[1] <- "train"

localH2O = h2o.init(max_mem_size = '3g', # use 6GB of RAM of *GB available
                    nthreads = -1) # use all CPUs (8 on my personal computer :3) 

train[,1] = as.factor(train[,1]) # convert digit labels to factor for classification
train_h2o = as.h2o(train)
test_h2o = as.h2o(test) 

s <- proc.time() 

model =
    h2o.deeplearning(x = 2:785,  # column numbers for predictors
                     y = 1,   # column number for label
                     training_frame = train_h2o, # data in H2O format
                     activation = "RectifierWithDropout", # algorithm
                     input_dropout_ratio = 0.2, # % of inputs dropout
                     hidden_dropout_ratios = c(0.5,0.5), # % for nodes dropout
                     balance_classes = TRUE, 
                     hidden = c(100,100), # two layers of 100 nodes
                     momentum_stable = 0.99,
                     nesterov_accelerated_gradient = T, # use it for speed
                     epochs = 15) # no. of epochs 

h2o.confusionMatrix(model) 

s - proc.time() 

h2o_y_test <- h2o.predict(model, test_h2o) 

df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "submission-r-h2o.csv", row.names=F) 

h2o.shutdown(prompt = F) 




rf <- randomForest(train, labels, xtest=test, ntree=numTrees)
predictions <- data.frame(ImageId=1:nrow(test), Label=levels(labels)[rf$test$predicted])
head(predictions) 

write_csv(predictions, "/home/mikhail/Desktop/GitProjects/KaggleCodes/digits/rf_benchmark.csv", 
          col_names = TRUE) 
 
 