# load the libraries
library(caret) 
library(readr)

# load the iris dataset
train <- read_csv("/home/mikhail/Desktop/GitProjects/KaggleCodes/digits/input/train.csv") 

# sample a subset   
numTrain <- 100 
rows <- sample(1:nrow(train), numTrain)
labels <- as.factor(train[rows,1])
train <- train[rows,-1] 

# define training control
train_control <- trainControl(method="cv", number=10)

# fix the parameters of the algorithm
grid <- expand.grid(.fL=c(0), .usekernel=c(FALSE))

# train the model
model <- train(labels~., data=train, trControl=trainControl(method = "cv", number = 10), method="bayesglm") 

# summarize results
print(model) 

