path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

library(data.table)
library(lubridate)
packages <- c("jsonlite")

library(purrr)
walk(packages,library,character.only=TRUE,warn.conflicts=TRUE)

traind <- fromJSON("train.json")
test <- fromJSON("test.json")

#unlist
vars <- setdiff(names(traind),c("photos","features"))
train <- map_at(traind, vars, unlist) %>% as.data.table()
test <- map_at(test,vars,unlist) %>% as.data.table()

head(train)
train[,.N/nrow(train),interest_level]
colSums(is.na(train))

#extract features and photos
photos <- train$photos
features <- train$features

photos_test <- test$photos
features_test <- test$features

train[,features := NULL]
train[,photos := NULL]

test[,features := NULL]
test[,photos := NULL]

#set column classes
d1 <- c("bathrooms","bedrooms","listing_id","price")
d2 <- c("building_id","manager_id","interest_level")
d3 <- c("latitude","longitude")

train[,(d1) := lapply(.SD, as.integer),.SDcols = d1]
train[,(d2) := lapply(.SD, as.factor),.SDcols = d2]
train[,(d3) := lapply(.SD, as.numeric),.SDcols = d3]

test[,(d1) := lapply(.SD, as.integer),.SDcols = d1]
test[,(d2[!(d2 %in% "interest_level")]) := lapply(.SD, as.factor),.SDcols = d2[!(d2 %in% "interest_level")]]
test[,(d3) := lapply(.SD, as.numeric),.SDcols = d3]

train[,created := as.POSIXct(created)]
test[,created := as.POSIXct(created)]

train[,created := NULL]
test[,created := NULL]



library(h2o)
localH2o <- h2o.init(nthreads = -1,max_mem_size = "20G")

trainh2o <- as.h2o(train)
testh2o <- as.h2o(test)

y <- "interest_level"
x <- setdiff(names(train),y)

gbm_one <- h2o.gbm(x = x
                   ,y = y
                   ,training_frame = trainh2o
                   ,ntrees = 100
                   ,distribution = "multinomial"
                   ,min_rows = 50)

h2o.shutdown()


library(xgboost)
train[,interest_level := as.character(interest_level)]
train[,interest_level := ifelse(interest_level == "low",0,ifelse(interest_level == "medium",1,2))]

dtrain <- xgb.DMatrix(data = data.matrix(train[,-c("description","display_address","street_address","listing_id","interest_level"),with=F]),label = train$interest_level)
dtest <- xgb.DMatrix(data = data.matrix(test[,-c("description","display_address","street_address","listing_id"),with=F]))


#default parameters
params <- list(
  booster = "gbtree",
  objective = "multi:softmax",
  num_class = 3,
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1,
  eval_metric = "merror" 
)

xgmodel_one <- xgb.train(params = params
                         ,data = dtrain
                         ,nrounds = 500
                         ,print_every_n = 10
                         ,maximize = F
                         ,verbose = 1)

pred <- predict(xgmodel_one,newdata = dtest)
pred_matrix <- matrix(pred, nrow = nrow(test),byrow = TRUE)


############################################################################
################NEW SOLUTION################################################
##############################################################################

rm(list=ls())

train_data <- fromJSON("train.json")
vars <- setdiff(names(train_data), c("photos","features"))
train_data <- map_at(train_data, vars, unlist) %>% as.data.table()

test_data <- fromJSON("test.json")
vars <- setdiff(names(test_data),c("photos","features"))
test_data <- map_at(test_data,vars,unlist)%>% as.data.table()

names(train_data)

word_features = c("building_id", "created", "description", "display_address", "street_address", "features", "listing_id", "manager_id", "photos")
train_X <- train_data[,-word_features,with=F]
test_X <- test_data[,-word_features,with=F]

train_X <- train_X[,-c("interest_level"),with=F]
train_Y <- ifelse(train_data$interest_level == "low",0, ifelse(train_data$interest_level == "medium",1,2))


set.seed(100)
pmt = proc.time()
model = xgboost(data = as.matrix(train_X), 
                label = train_Y,
                eta = 0.3,
                max_depth = 6, 
                nround=500, 
                subsample = 1,
                colsample_bytree = 1,
                seed = 100,
                eval_metric = "merror",
                objective = "multi:softprob",
                num_class = 3,
                missing = NaN,
                silent = 1)
show(proc.time() - pmt)


pred <- predict(model, as.matrix(test_X), missing = NaN)
pred_matrix <- matrix(pred, nrow = nrow(test_data), byrow = T)
pred_submission <- cbind(test_data$listing_id, pred_matrix)
colnames(pred_submission) <- c("listing_id","low","medium","high")
pred_df <- as.data.table(pred_submission)
write.csv(pred_df,"xgboost_base_one.csv",row.names = F) #0.67924


#Tune Hyperparameters

library(caTools)
xd <- sample.split(train_data$interest_level,SplitRatio = 0.6)
train_XX <- train_data[xd]
train_XX <- train_XX[,-word_features,with=F]
train_XX[,interest_level := ifelse(train_XX$interest_level == "low",0, ifelse(train_XX$interest_level == "medium",1,2))]

valid_XX <- train_data[!(xd)]
valid_XX <- valid_XX[,-word_features,with=F]
valid_XX[,interest_level := ifelse(valid_XX$interest_level == "low",0, ifelse(valid_XX$interest_level == "medium",1,2))]

set.seed(100)
pmt = proc.time()
xgb_params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eta=0.1,
  gamma=5,
  max_depth=4,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1,
  eval_metric = "mlogloss",
  num_class = 3
)

dtrain <- xgb.DMatrix(data = data.matrix(train_XX[,-c("interest_level"),with=F]),label = train_XX$interest_level)
dval <- xgb.DMatrix(data = data.matrix(valid_XX[,-c("interest_level"),with=F]),label = valid_XX$interest_level)

gbdt <- xgb.train(params = xgb_params
                  ,data = dtrain
                  ,nrounds = 500
                  ,watchlist = list(train = dtrain, val = dval)
                  ,print_every_n = 10
                  ,maximize = FALSE)

#local validation = 0.670018


df_train <- xgb.DMatrix(data = data.matrix(train_X),label = train_Y)
gbdt_full <- xgboost(data = as.matrix(train_X)
                     ,label = train_Y
                     ,params = xgb_params
                     ,nrounds = 500
                     ,print_every_n = 10
                     ,maximize = F)

pred <- predict(gbdt_full,as.matrix(test_X),missing = NaN)
pred_matrix <- matrix(pred, nrow(test_data), byrow = T)
pred_submission <- cbind(test_data$listing_id, pred_matrix)
colnames(pred_submission) <- c("listing_id","low","medium","high")
pred_df <- as.data.table(pred_submission)
write.csv(pred_df,"xgboost_tuned_two.csv",row.names = F)


#Tune Hyperparameters

library(caTools)
xd <- sample.split(train_data$interest_level,SplitRatio = 0.6)
train_XX <- train_data[xd]
train_XX <- train_XX[,-word_features,with=F]
train_XX[,interest_level := ifelse(train_XX$interest_level == "low",1, ifelse(train_XX$interest_level == "medium",2,3))]
train_XX[,interest_level := as.factor(interest_level)]


valid_XX <- train_data[!(xd)]
valid_XX <- valid_XX[,-word_features,with=F]
#valid_XX[,interest_level := ifelse(valid_XX$interest_level == "low",0, ifelse(valid_XX$interest_level == "medium",1,2))]
valid_XX[,interest_level := as.factor(interest_level)]

library(mlr)
traintask <- makeClassifTask(data = train_XX,target = "interest_level")
validtask <- makeMultilabelTask(data = valid_XX,target = "interest_level")

#create learner
lrn <- makeLearner("classif.xgboost",predict.type = "response")
lrn$par.vals <- list(
  objective = "multi:softprob",
  eval_metric = "merror",
  nrounds = 500L,
  eta = 0.1,
  maximize = FALSE
)

getParamSet(lrn)

#set parameters
params <- makeParamSet(
  makeIntegerParam("max_depth",lower = 3,upper = 10),
  makeNumericParam("subsample",lower = 0.3,upper = 0.9),
  makeNumericParam("colsample_bytree",lower = 0.3,upper = 0.9)
)

#set CV
rdesc <- makeResampleDesc(method = "Holdout")

#search strategy
ctrl <- makeTuneControlRandom(maxit = 10L)

#set parallel backend
library(parallelMap)
library(parallel)
parallelStartMulticore(cpus = detectCores())

mytune <- tuneParams(learner = lrn,task = traintask,resampling = rdesc,measures = logloss, par.set = params,control = ctrl,show.info = F)

yeast <- getTaskData(yeast.task)


trainYYY <- train_XX[,.(interest_level)]
dummy <- trainYYY[1:10]

dcast.data.table(data = dummy,formula =  .~ interest_level,)


