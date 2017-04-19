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

#create data for train
train_id <- train$building_id
test_id <- test$building_id

#encode target variable
#medium - 2, low - 1, high - 0


# Remove charcter variables -----------------------------------------------

train_label <- train$interest_level
train_label <- as.integer(as.factor(train_label))-1

X_train <- train[,c("building_id","description","features","created","photos","interest_level") := NULL]

char <- colnames(X_train)[sapply(X_train, is.character)]
X_train[,(char) := NULL]
X_test <- test[,colnames(X_train),with=F]


#use baseline model
library(xgboost)

d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
d_test <- xgb.DMatrix(data = as.matrix(X_test))

#default 
params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eta=0.3,
  gamma=0,
  max_depth=6,
  min_child_weight=1,
  subsample=1,
  colsample_bytree=1,
  eval_metric = "mlogloss",
  num_class = 3
)

xgbcv <- xgb.cv(params = params
                ,data = d_train
                ,nrounds = 500
                ,nfold = 5
                ,showsd = T
                ,stratified = T
                ,print_every_n = 10
                ,early_stop_round = 5
                ,maximize = F
)
dx <- which.min(xgbcv$evaluation_log$test_mlogloss_mean)
xgbcv$evaluation_log$test_mlogloss_mean[129] #0.6457
#

xgb_model <- xgb.train(params = params
                       ,data = d_train
                       ,nrounds = dx
                       ,watchlist = list(train = d_train)
                       ,maximize = F
                       )

xgb_pred <- predict(xgb_model, d_test)
allpredictions <- as.data.table(matrix(predict(xgb_model, d_test), nrow = dim(test), byrow = T))
allpredictions <- cbind(allpredictions, test$listing_id)
names(allpredictions) <- c("high","low","medium","listing_id")

# create submission file --------------------------------------------------
fwrite(allpredictions, "one_baseline_march9.csv") #0.64 


# Now tune the model and set CV -------------------------------------------

xgb_model <- function(param, nround){
  print(paste(names(param),param,collapse = " , "))
  
  #select number of rounds
  bst_cv <- xgb.cv(params = param
                   ,data = d_train
                   ,nrounds = nround
                   ,nfold = 5
                   ,showsd = T
                   ,stratified = T
                   ,print_every_n = 10
                   ,early_stop_round = 5
                   ,maximize = F
  )
  nround_sel <- which.min(bst_cv$evaluation_log$test_mlogloss_mean)
  print(paste("nround selected is :", nround_sel));
  print(bst_cv$evaluation_log$test_mlogloss_mean[nround_sel]);
  
  #train final model
  final_model <- xgb.train(params = param
                           ,data = d_train
                           ,nrounds = nround_sel
                           ,watchlist = list(train = d_train)
                           ,maximize = F
  )
  return (final_model)
  
}

params1 <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eta=0.1,
  gamma=0.1,
  max_depth=15,
  min_child_weight=10,
  subsample=0.8,
  colsample_bytree=0.8,
  eval_metric = "mlogloss",
  num_class = 3
)

xgb_tuned_1 <- xgb_model(params1, 200)

allpredictions <- as.data.table(matrix(predict(xgb_tuned_1, d_test), nrow = dim(test), byrow = T))
allpredictions <- cbind(allpredictions, test$listing_id)
names(allpredictions) <- c("high","low","medium","listing_id")


# create submission ile ---------------------------------------------------
fwrite(allpredictions, "two_baseline_tuning_march10.csv")


#create model matrix for charcter variables
train_matrix <- model.matrix(~.+0, data = X_train[,char,with=F])
X_train[,(char) := NULL]

X_train <- cbind(X_train, train_matrix)
saveRDS(X_train,"X_train.rds")




###############New Feature Enginnering - March 10########################

train_label <- train$interest_level
train_label <- as.integer(as.factor(train_label))-1

test_id <- test$listing_id


# Feature Engineering -----------------------------------------------------
# Create features in train data -------------------------------------------

library(stringr)

X_train <- copy(train)[,c("description","features","photos","listing_id","created","interest_level","building_id","manager_id","display_address") := NULL]
X_test <- copy(test)[,c("description","features","photos","listing_id","created","building_id","manager_id","display_address") := NULL]

train[,created := ymd_hms(created)]
test[,created := ymd_hms(created)]

# train[,created_year := year(created)]
# train[,created_year := NULL]

train[,created_month := month(created)]

train[,created_day := day(created)]

train[,created_hour := hour(created)]
test[,created_hour := hour(created)]

train[,features_count := lapply(features, function(x) length(x))]
test[,features_count := unlist(lapply(features, function(x) length(x)))]

train[,photos_count := lapply(photos, function(x) length(x))]
test[,photos_count := unlist(lapply(photos, function(x) length(x)))]

train[,building_id_integer := as.integer(as.factor(building_id))]
test[,building_id_integer := as.integer(as.factor(building_id))]

train[,manager_id_integer := as.integer(as.factor(manager_id))]
test[,manager_id_integer := as.integer(as.factor(manager_id))]


# train[,street_address_number := str_extract(string = street_address,pattern = "^\\d+")]
# 
# train[,street_address_number := as.integer(street_address_number)]

#create a one hot encoding for building id and manager id
# one_hot <- train[,.(building_id,manager_id)]
# one_hot_test <- test[,.(building_id,manager_id)]
# 
# one_hot_encode <- model.matrix(~.+0, data = one_hot)
# saveRDS(one_hot_encode, "building_manager_one_hot.rds")
# 
# one_hot_encode <- model.matrix(~.+0, data = one_hot_test)
# saveRDS(one_hot_encode, "building_manager_one_hot_test.rds")

#one hot is ememory inefficient on high cardinality variables

# #join matrix in X_train
# X_train <- cbind(X_train,one_hot_encode)
# X_test <- cbind(X_test,one_hot_encode)



# Prepare data for xgboost ------------------------------------------------
library(xgboost)

d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)

params <- list(
  booster = "gbtree",
  objective = "multi:softprob",
  eta=0.1,
  gamma=0,
  max_depth=4,
  min_child_weight=1,
  subsample=0.7,
  colsample_bytree=0.7,
  eval_metric = "mlogloss",
  num_class = 3
)

#us XGB to check CV accuracy
xgb_tune <- function(param, nround){
  print(paste(names(param),param,collapse = " , "))
  
  #select number of rounds
  bst_cv <- xgb.cv(params = param
                   ,data = d_train
                   ,nrounds = nround
                   ,nfold = 5L
                   ,stratified = T
                   ,print_every_n = 10
                   ,early_stopping_rounds = 5
                   ,maximize = F)
  
  nround_sel <- which.min(bst_cv$evaluation_log$test_mlogloss_mean)
  print(paste("nround selected is:", nround_sel))
  print(bst_cv$evaluation_log$test_mlogloss_mean[nround_sel])

  #train final model
  final_model <- xgb.train(params = param
                           ,data = d_train
                           ,nrounds = nround_sel
                           ,watchlist = list(train = d_train)
                           ,maximize = F
  )
  return (final_model)
  
}


# Tuning ------------------------------------------------------------------
#With 5 raw variables
xgb1 <- xgb_tune(params, 500) #0.6510

#with 5 + 1 variables (created_month)
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.6524

#
X_train[,created_month := NULL]
X_train[,created_day := train$created_day] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.6522

#score improved - so kept the variable
X_train[,created_day := NULL]
X_train[,created_hour := train$created_hour] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.6262

#score improved - so kept the variable
#X_train[,created_hour := NULL]
X_train[,features_count := unlist(train$features_count)] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.6051

#score improved - so kept the variable
X_train[,photos_count := unlist(train$photos_count)] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500)#0.5934

#score improved - so kept the variable
X_train[,building_id_integer := train$building_id_integer] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500)#0.5784

#score improved - so kept the variable
X_train[,manager_id_integer := train$manager_id_integer] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.5682

#this variable needs to be removed
X_train[,street_address_number := train$street_address_number] 
d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
xgb2 <- xgb_tune(params, 500) #0.5692

#prepare test data with selected variables
X_test <- cbind(X_test,test[,.(created_hour,features_count,photos_count,building_id_integer,manager_id_integer)])

d_train <- xgb.DMatrix(data = as.matrix(X_train),label = train_label)
d_test <- xgb.DMatrix(data = as.matrix(X_test))

#train model
xgboot <- xgb_tune(params,500) #this gave 0.5685 as CV error


# Create predictions ------------------------------------------------------
allpredictions <- as.data.table(matrix(predict(xgboot, d_test), nrow = dim(test), byrow = T))
allpredictions <- cbind(allpredictions, test$listing_id)
names(allpredictions) <- c("high","low","medium","listing_id")


# create submission ile ---------------------------------------------------
fwrite(allpredictions, "three_xgboost_with_features.csv")






















