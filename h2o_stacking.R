library(h2o)

h2o.init(nthreads = -1,max_mem_size = "20G")

trainh2o <- as.h2o(train_one)
testh2o <- as.h2o(test_one)

y <- "interest_level"
x <- setdiff(colnames(train_one), c(y,"listing_id","medium","latitude"))

clf_glm <- h2o.glm(x = x
                   ,y = y
                   ,training_frame = trainh2o
                   ,nfolds = 5L
                   ,seed = 1
                   ,keep_cross_validation_predictions = T
                   ,family = 'multinomial'
                   ,alpha = 0.7
                   ,lambda_search = T
#                   ,standardize = T
                   )

h2o.varimp(clf_glm)

clf_train <- as.data.table(h2o.cross_validation_holdout_predictions(clf_glm))
clf_train <- data.table(listing_id = train_one$listing_id, gl_high = clf_train$high, gl_low = clf_train$low, gl_medium = clf_train$medium)


clf_test <- as.data.table(h2o.predict(clf_glm, testh2o))
clf_test <- data.table(listing_id = test_one$listing_id, gl_high = clf_test$high, gl_low = clf_test$low, gl_medium = clf_test$medium)


clf_dl <- h2o.deeplearning(x = x
                           ,y = y
                           ,training_frame = trainh2o
                           ,nfolds = 5L
                           ,keep_cross_validation_predictions = T
                           ,variable_importances = T
                           ,hidden = c(100,100)
                           ,l2 = 0.0001
                           )
h2o.varimp_plot(clf_dl)

clf_pred <- as.data.table(h2o.predict(clf_dl,testh2o))

dl_train <- h2o.cross_validation_holdout_predictions(clf_dl)
dl_train <- as.data.table(dl_train)

dl_train <- data.table(listing_id = train_one$listing_id, dl_high = dl_train$high, dl_low = dl_train$low, dl_medium = dl_train$medium)
dl_test <- data.table(listing_id = test_one$listing_id, dl_high = clf_pred$high, dl_low = clf_pred$low, dl_medium = clf_pred$medium)


clf_rf <- h2o.randomForest(x = x
                           ,y = y
                           ,training_frame = trainh2o
                           ,keep_cross_validation_predictions = T
                           ,nfolds = 5L
                           ,ntrees = 500
                           ,max_depth = 6L
                           ,min_rows = 1
                           ,sample_rate = 0.025)
h2o.varimp_plot(clf_rf)

rf_train <- as.data.table(h2o.cross_validation_holdout_predictions(clf_rf))
rf_train <- data.table(listing_id = train_one$listing_id, rf_high = rf_train$high, rf_low=rf_train$low, rf_medium=rf_train$medium)


rf_pred <- as.data.table(h2o.predict(clf_rf, testh2o))
rf_test <- data.table(listing_id = test_one$listing_id, rf_high = rf_pred$high, rf_low = rf_pred$low, rf_medium = rf_pred$medium)


clf_gbm <- h2o.gbm(x = x
                   ,y = y
                  ,training_frame = trainh2o
                  ,nfolds = 5L
                  ,keep_cross_validation_predictions = T
                  ,ntrees = 500
                  ,max_depth = 6L
                  ,learn_rate = 0.025)

h2o.varimp_plot(clf_gbm)

gbm_train <- as.data.table(h2o.cross_validation_holdout_predictions(clf_gbm))
clf_pred <- as.data.table(h2o.predict(clf_gbm, testh2o))


gbm_train <- data.table(listing_id = train_one$listing_id, gbm_high = gbm_train$high, gbm_low=gbm_train$low, gbm_medium = gbm_train$medium)
gbm_test <- data.table(listing_id = test_one$listing_id, gbm_high = clf_pred$high, gbm_low = clf_pred$low, gbm_medium = clf_pred$medium)


#SVM
library(caret)

train_one[is.na(manager_level_high), manager_level_high := 0.0772]
train_one[is.na(manager_level_low), manager_level_low := 0.6938]
train_one[is.na(manager_level_medium), manager_level_medium := 0.2290]

test_one[is.na(manager_level_high), manager_level_high := 0.0772]
test_one[is.na(manager_level_low), manager_level_low := 0.6938]
test_one[is.na(manager_level_medium), manager_level_medium := 0.2290]


ctrl <- trainControl(method = 'cv', savePredictions = T,number = 5L,classProbs = T)
mod <- train(interest_level ~ ., data = train_one[,-c('medium','listing_id','latitude'),with=F], method = 'svmLinear', trControl = ctrl)

mod$pred


mod_pred <- predict(mod, test_one,type='prob')
head(mod_pred)

svm_train <- data.table(listing_id = train_one$listing_id, svm_high=mod$pred$high, svm_low=mod$pred$low, svm_medium=mod$pred$medium)
svm_test <- data.table(listing_id = test_one$listing, svm_high=mod_pred$high, svm_low=mod_pred$low,svm_medium=mod_pred$medium)



mega_train <- data.table(listing_id = train_one$listing_id,interest_level = train_one$interest_level)
mega_test <- data.table(listing_id = test_one$listing_id)


mega_train <- clf_train[mega_train,on='listing_id']
mega_test <- clf_test[mega_test,on='listing_id']

mega_train <- dl_train[mega_train,on='listing_id']
mega_test <- dl_test[mega_test,on='listing_id']

mega_train <- rf_train[mega_train, on='listing_id']
mega_test <- rf_test[mega_test, on='listing_id']

mega_train <- gbm_train[mega_train, on='listing_id']
mega_test <- gbm_test[mega_test, on='listing_id']

mega_train <- svm_train[mega_train, on='listing_id']
mega_test <- svm_test[mega_test, on='listing_id']

mega_train[,c('high','low','price') := list(train_one$high, train_one$low, train_one$price)]
mega_test[,c('high','low','price') := .(test_one$high, test_one$low, test_one$price)]

fwrite(mega_train,"mega_train.csv")
fwrite(mega_test,"mega_test.csv")

save.image("stacking_data.RData")
load("/home/manish/Desktop/Assignment/assignment1/stacking_data.RData")


#level 2 model - extra trees
# medium = 2, low = 1, high = 0
library(xgboost)
library(caTools)
d_train <- xgb.DMatrix(data = as.matrix(mega_train[,-c('listing_id','interest_level'),with=F]), label=target)
d_test <- xgb.DMatrix(data = as.matrix(mega_test[,-c('listing_id'),with=F])) 


splitX <- sample.split(Y = mega_train$interest_level, SplitRatio = 0.6)

dX_train <- xgb.DMatrix(data = as.matrix(mega_train[splitX,-c('listing_id','interest_level'),with=F]), label=target[splitX])
dX_val <- xgb.DMatrix(data = as.matrix(mega_train[!splitX,-c('listing_id','interest_level'),with=F]), label=target[!splitX])

dim(dX_train)
dim(dX_val)

param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              alpha=32,
              #gamma = 0,
              max_depth = 6,
              min_child_weight = 0,
              subsample = 1,
              colsample_bytree = 1
)

cv_clf <- xgb.cv(params = param
                 ,data = d_train
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 10
                 ,maximize = F
)
#0.415608+

set.seed(2016010)

watch <- list(val= dX_val, train=dX_train)

xgb2 <- xgb.train(data = d_train,
                  params = param,
                  watchlist=watch,
                  early_stopping_rounds = 10,
                  maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 1000,
                  print_every_n = 10
                  
)

sPreds <- as.data.table(t(matrix(predict(xgb2, d_test), nrow=3, ncol=nrow(d_test))))

colnames(sPreds) <- c("high","low","medium")

sPreds <- data.table(listing_id=test$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "stack_tunedxgb.csv") #0.78201


#Bagging with different seeds
all_class = {}
for (seed in c(1960,2018,1988,1:7))
{
  print (seed)
  for (subsample in c(0.2,0.4,0.6,0.8))
  {
    print (subsample)
    param <- list(  objective           = "multi:softprob", 
                    num_class           = 3,
                    #max_delta_step=8,
                    booster             = "gbtree",
                    eta                 = 0.1,
                    max_depth           = 6, #7
                    alpha=32,
                    min_child_weight = 0,
                    subsample           = subsample,
                    colsample_bytree    = 1
    )
    
    print(paste("Now training the model with",seed,"and",subsample))
    set.seed(seed)
    clf2 <- xgb.train(   params              = param, 
                         data                = d_train,
                         nrounds             = 500, 
                         verbose             = 0,
                         early_stopping_rounds    = 20,
                         watchlist           = list(val=dX_val,train=d_train),
                         maximize            = FALSE,
                         eval_metric       = "mlogloss"
    )
    
    pred_exp2 = t(matrix(predict(clf2, d_test), nrow=3, ncol=nrow(d_test)))
    colnames(pred_exp2) <- c("high","low","medium")
    print(head(all_class))
    all_class = cbind(all_class,pred_exp2)
  }}

dim(all_class)
class(all_class)

all_class <- as.data.table(all_class)

ah <- colnames(all_class)[seq(1,120,3)]
al <- colnames(all_class)[seq(2,120,3)]
am <- colnames(all_class)[seq(3,120,3)]

j_high <- all_class[,seq(1,120,3),with=F]
j_low <- all_class[,seq(2,120,3),with=F]
j_medium <- all_class[,seq(3,120,3),with=F]

j_high[,high_mean := rowMeans(.SD)]
j_low[,low_mean := rowMeans(.SD)]
j_medium[,medium_mean := rowMeans(.SD)]


#final submission
submit <- data.table(listing_id = test_one$listing_id, high = j_high$high_mean, medium = j_medium$medium_mean, low = j_low$low_mean)
fwrite(submit, "stack_bagging_xgboost.csv")


#h2o






# path <- "/home/manish/Desktop/MyData/Data/Amonth_Wise/November 2016/SocialCops/file/checker/"
# setwd(path)
# 
# t1 <- fread("mytest1.csv")
# head(t1)
# 
# t2 <- fread("test.csv")
# head(t2)
# head(t2$id,25)
# 
# zx <- str_detect(string = t2$id[1:30], pattern = ".*\\d+")
# length(zx)
# 
# str_detect(string = t2$id[1:30], pattern = ".*\\d+")
# zx <- str_extract_all(string = t2$id, pattern = '^\\d+$',simplify = T)
# zx <- grep(pattern = "^\\d+$",x = t2$id,value = T)
# 
# t3 <- fread("train.csv")
# 
# head(t3$id,30)
# grep(pattern = "^\\d+$", x = t3$id, value = T)
# 
# t3[,uniqueN(id)]
# t2[,uniqueN(id)]
# 
# asd <- sample(x = paste("it",1998:19999998,sep = ""),size = nrow(t3),replace = F)
# 
# paste(paste("a",sample(1:9,2,replace = F),sep = ""), paste("q",sample(1:9,2,replace = F),sep = ""),paste("z",sample(1:9,2,replace = F),sep = ""),sep = "")
# 
# 
# x1 <- sample(x = paste(sample(letters[1:24],2,replace = T),sample(1:24000,2,replace = T),sep = ""),size = 10,replace = T)
# x1
# 
# x2 <- sample(x = paste(letters[10:24],sample(100:200,2,replace = T),sep = ""),size = nrow(t3),replace = T)
# x2
# 
# cv <- paste0(x1,x2)
# length(unique(cv))
# length(cv)
# 
# post <- fread("/home/manish/Desktop/check_post.csv")
# head(post)
# sapply(post, class)
# post[,score := score*runif(n = nrow(post),min = 1,max = 5)]
# post[,score := score*1.03]
# 
# fwrite(post,"check_post_two.csv")














