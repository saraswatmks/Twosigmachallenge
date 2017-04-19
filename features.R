path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

packages <- c("jsonlite","purrr","data.table","xgboost")
purrr::walk(packages, library, character.only=TRUE, warn.conflicts = F)

train <- fromJSON("train.json")
test <- fromJSON("test.json")

#Train
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)

#Test
vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test$listing_id

setDT(train)
setDT(test)

library(ggplot2)
library(lubridate)
library(e1071)
library(stringr)
ggplot(train,aes(manager_id))+geom_bar()

join <- rbindlist(list(train,test),fill=TRUE)

#create simple features
join[,feature_count := unlist(lapply(features, length))]
join[,desc_count := str_count(description)]
join[,photos_count := unlist(lapply(photos, length))]
head(join)

join[,c("photos","features","description") := NULL]

join[,created := ymd_hms(created)]
join[,xmonth := month(created)]
join[,xday := day(created)]
join[,xhour := hour(created)]

join[,created := NULL]

library(RecordLinkage)
join[,sim_address := levenshteinSim(display_address, street_address)]
join[is.na(sim_address), sim_address := 0.0]

join[,c("display_address","street_address") := NULL]

library(caret)
lapply(join[,.(bathrooms,bedrooms,price)], skewness)

join[,.N,bathrooms]
join[bathrooms == 112.0, bathrooms := 10.0]
join[bathrooms == 112.0, bathrooms := 10.0]
join[bathrooms == 20.0, bathrooms := 10.0]
join[bathrooms == 10.0, bathrooms := 1.0]

join[,.N,price]
join[order(price),.N]
join[,summary(price)]

join[price == 4490000.000, price := 1150000]
ggplot(join,aes(price))+geom_density()+scale_x_log10()
join[,price := log10(price+1)]

join[,building_id := as.integer(as.factor(building_id))]
join[,manager_id := as.integer(as.factor(manager_id))]

train_one <- join[1:nrow(train)]
test_one <- join[(nrow(train)+1):124011]

train_one[,interest_level := as.factor(interest_level)]

# mulmode <- multinom(interest_level ~.-listing_id,data = train_one)
# mulmode
# 
# mulpred <- predict(mulmode, test_one)
# mulpred[1:20]

gbmone <- gbm::gbm(interest_level ~ .-listing_id,distribution = "multinomial",data = train_one,n.trees = 500,interaction.depth = 2,n.minobsinnode = 10,train.fraction = 0.5,cv.folds = 5L)
gbmone$cv.fitted

gbmpred <- as.data.table(t(matrix(predict(gbmone, test_one), nrow=3, ncol=nrow(test_one))))


loi <- knn3(x = as.matrix(train_one[,-c("listing_id","interest_level"),with=F]),y = train_one$interest_level,k = 5)
loiped <- predict(loi, train_one[,-c("listing_id","interest_level"),with=F])
loiped212 <- predict(loi,test_one[,-c('listing_id','interest_level'),with=F])



# Bind kNN prediction -----------------------------------------------------

train_one <- cbind(train_one, loiped)
test_one <- cbind(test_one, loiped212)

test_one[,interest_level := NULL]

library(xgboost)

target <- as.integer(as.factor(train_one$interest_level))-1
dtrain <- xgb.DMatrix(data = as.matrix(train_one[,-c('listing_id','interest_level'),with=F]), label=target)
dtest <- xgb.DMatrix(data = as.matrix(test_one[,-c('listing_id'),with=F]))


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
              subsample = 1,
              colsample_bytree = 1
)

cv_clf <- xgb.cv(params = param
                 ,data = dtrain
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 10
                 ,maximize = F
)

clf_xgb <- xgb.train(params = param
                     ,data = dtrain
                     ,nrounds = 180
                    ,watchlist = list(train = dtrain)
                    ,print_every_n = 10
                    )

imp <- xgb.importance(feature_names = colnames(train_one),model = clf_xgb)
xgb.plot.importance(importance_matrix = imp,top_n = 20)

sPreds <- as.data.table(t(matrix(predict(clf_xgb, dtest), nrow=3, ncol=nrow(dtest))))

colnames(sPreds) <- c("high","low","medium")

sPreds <- data.table(listing_id=test_one$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "knn_feature_xgboost.csv") #1.27


#ensemble xgboost and knn
sol1 <- fread("brandonsubmission.csv")

#average
ensem <- data.table(listing_id = sol1$listing_id, high = (sol1$high + sPreds$high)/2, medium = (sol1$medium + sPreds$medium)/2, low = (sol1$low + sPreds$low)/2)
ensem
fwrite(ensem, "knn_xgboost_aver.csv") #0.58042


#weighted_average
wei_ensem <- data.table(listing_id = sol1$listing_id, high = (0.7*(sol1$high) + 0.3*(sPreds$high))/2, medium = (0.7*(sol1$medium) + 0.3*(sPreds$medium))/2, low = (0.7*(sol1$low) + 0.3*(sPreds$low))/2)
wei_ensem
fwrite(wei_ensem,"knn_xgboost_weightensem.csv") #0.55319

#bind gbm prediction
xd <- gbmone$cv.fitted
colnames(xd) <- c("V1","V2","V3")

train_one <- cbind(train_one,xd)
test_one <- cbind(test_one,gbmpred)

dtrain <- xgb.DMatrix(data = as.matrix(train_one[,-c('listing_id','interest_level'),with=F]), label=target)
dtest <- xgb.DMatrix(data = as.matrix(test_one[,-c('listing_id'),with=F]))


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
              subsample = 1,
              colsample_bytree = 1
)

cv_clf <- xgb.cv(params = param
                 ,data = dtrain
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 10
                 ,maximize = F
) #0.414840 CV



clf_xgb <- xgb.train(params = param
                     ,data = dtrain
                     ,nrounds = 193
                     ,watchlist = list(train = dtrain)
                     ,print_every_n = 10
)

imp <- xgb.importance(feature_names = colnames(dtrain),model = clf_xgb)
xgb.plot.importance(importance_matrix = imp,top_n = 20)


sPreds <- as.data.table(t(matrix(predict(clf_xgb, dtest), nrow=3, ncol=nrow(dtest))))

colnames(sPreds) <- c("high","low","medium")

sPreds <- data.table(listing_id=test_one$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "knn_gbm_feature_xgboost.csv") #1.28705


clf_svm <- svm(interest_level ~ .-listing_id, data = train_one, probability =  TRUE)
clf_svm

svm_pred <- predict(clf_svm,newdata = as.matrix(test_one))
svm_pred[1:10]

#add manager_train.csv and manager_test.csv to the model above.
manager_train <- fread("manager_train.csv")
manager_test <- fread("manager_test.csv")

# train_one <- cbind(train_one,manager_train)
# test_one <- cbind(test_one,manager_test)

train_one <- manager_train[train_one, on="listing_id"]
test_one <- manager_test[test_one, on='listing_id']

str(train_one)
str(test_one)

dtrain <- xgb.DMatrix(data = as.matrix(train_one[,-c('listing_id','interest_level','V3','medium'),with=F]), label=target)
dtest <- xgb.DMatrix(data = as.matrix(test_one[,-c('listing_id','V3','medium'),with=F]))


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.025,
              alpha = 2,
              #lambda = 4,
              max_depth = 6,
              min_child_weight = 5,
              subsample = 0.7,
              colsample_bytree = 0.7
)

cv_clf <- xgb.cv(params = param
                 ,data = dtrain
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 20
                 ,maximize = F
)
#0.416751 CV, eta = 0.1 and all default
#0.421662, all same, depth = 8
#0.429977, all same, depth = 10
#0.417065, alpha = 2
#0..420318 alpha = 32
#0.417605 alpha = 4
#0.416992 min_child_ = 0
#0.416453 min_child = 5

clf_xgb <- xgb.train(params = param
                     ,data = dtrain
                     ,nrounds = 182
                     ,watchlist = list(train = dtrain)
                     ,print_every_n = 10
)

imp <- xgb.importance(feature_names = colnames(dtrain),model = clf_xgb)
xgb.plot.importance(importance_matrix = imp,top_n = 20)



############################################################# 


# join[, high := as.integer(interest_level == "high")]
# join[, low := as.integer(interest_level == "low")]
# join[, medium := as.integer(interest_level == "medium")]
# 
# join[,high_mean := mean(high,na.rm=T),manager_id]
# join[,low_mean := mean(low,na.rm=T),manager_id]
# join[,medium_mean := mean(medium, na.rm=T), manager_id]
# 
# join[,high_sd := sd(high, na.rm = T),manager_id]
# join[,low_sd := sd(low, na.rm = T),manager_id]
# join[,medium_sd := sd(medium, na.rm = T),manager_id]

join[,bedroom_mean := mean(bedrooms), manager_id]
join[,bathrooms_mean := mean(bathrooms), manager_id]
join[,price_mean := mean(price), manager_id]

join[,bedroom_sd := sd(bedrooms), manager_id]
join[,bathroom_sd := sd(bathrooms),manager_id]
join[,price_sd := sd(price), manager_id]

join[,bathroom_median := median(bathrooms),manager_id]
join[,bedroom_median := median(as.numeric(bedrooms)), manager_id]
join[,price_median := median(price),manager_id]


join[,bbedroom_mean := mean(bedrooms), building_id]
join[,bbathrooms_mean := mean(bathrooms), building_id]
join[,bprice_mean := mean(price), building_id]

join[,bbedroom_sd := sd(bedrooms), building_id]
join[,bbathroom_sd := sd(bathrooms), building_id]
join[,bprice_sd := sd(price), building_id]

join[,bbathroom_median := median(bathrooms),manager_id]
join[,bbedroom_median := median(as.numeric(bedrooms)), manager_id]
join[,bprice_median := median(price),manager_id]

join[,c("bathroom_median","bedroom_median") := NULL]

join[,x1 := (bathrooms - mean(bathrooms))/sd(bathrooms), manager_id]
join[,x2 := (bedrooms - mean(bedrooms))/sd(bedrooms),manager_id]
join[,x3 := (price - mean(price))/sd(price),manager_id]

join[,x4 := (bathrooms - mean(bathrooms))/sd(bathrooms), building_id]
join[,x5 := (bedrooms - mean(bedrooms))/sd(bedrooms), building_id]
join[,x6 := (price - mean(price))/sd(price), building_id]


#prepare data for modeling
new_join <- join[,-c("listing_id","high","low","medium","interest_level"),with=F]

dead <- c("high_mean","low_mean","medium_mean","high_sd","low_sd","medium_sd","high_median")
batch_one <- setdiff(colnames(new_join), dead)

X_train <- new_join[1:49352,batch_one,with=F]
X_test <- new_join[49353:124011,batch_one,with=F]

target <- train$interest_level
target <- as.integer(as.factor(target))-1
target[1:10] # medium = 2, low = 1, high = 0

d_train <- xgb.DMatrix(data = as.matrix(X_train), label=target, missing = NA)
#d_val <- xgb.DMatrix(data = as.matrix(train_one[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(X_test),missing = NA)


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
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

set.seed(201609)

watch <- list(val= d_val, train=d_train)

xgb2 <- xgb.train(data = d_train,
                  params = param,
                  #watchlist=watch
                  #early_stopping_rounds = 10
                  #maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 329,
                  print_every_n = 10
                  
)

xgb.plot.deepness(model = xgb2)
xgb.plot.importance(importance_matrix = xgb.importance(colnames(X_train), model = xgb2),top_n = 30)


sPreds <- as.data.table(t(matrix(predict(xgb2, d_test), nrow=3, ncol=nrow(d_test))))

colnames(sPreds) <- c("high","low","medium")

sPreds <- data.table(listing_id=test$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "feature_attempt_xgboost.csv")


##############################################################
####17th April 2017##########################################







#add manager_train.csv and manager_test.csv to the model above.
manager_train <- fread("manager_train.csv")
manager_test <- fread("manager_test.csv")

# train_one <- cbind(train_one,manager_train)
# test_one <- cbind(test_one,manager_test)

train_one <- join[1:nrow(train)]
test_one <- join[(nrow(train)+1):124011]

train_one <- manager_train[train_one, on="listing_id"]
test_one <- manager_test[test_one, on='listing_id']

train_one[is.na(train_one)] <- -1
test_one[is.na(test_one)] <- -1

train$interest_level[1:10]
train_one[,interest_level := as.integer(as.factor(interest_level))-1]
train_one$interest_level[1:10] #high = 0, low = 1, medium = 2
head(train_one)

d_train <- xgb.DMatrix(data = as.matrix(train_one[,-c('listing_id','interest_level'),with=F]), label=train_one$interest_level, missing = -1)
#d_val <- xgb.DMatrix(data = as.matrix(train_one[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(test_one[,-c('interest_level','listing_id')]),missing = -1)


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
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
                 ,prediction = T
)



#now using CV predictions from xgb model and and creating values for manager_id
cv_pred <- cv_clf$pred
cv_pred <- as.data.table(cv_pred)

colnames(cv_pred) <- c('high_cv','low_cv','medium_cv')

train_one <- cbind(train_one,cv_pred)

bond_dat <- rbindlist(list(train_one,test_one),fill = T)

bond_dat[,man_cv_1 := mean(high_cv,na.rm = T),manager_id]
bond_dat[,man_cv_2 := mean(low_cv,na.rm = T),manager_id]
bond_dat[,man_cv_3 := mean(medium_cv,na.rm = T),manager_id]

bond_dat[,man_cv_4 := mean(high_cv,na.rm = T),building_id]
bond_dat[,man_cv_5 := mean(low_cv,na.rm = T),building_id]
bond_dat[,man_cv_6 := mean(medium_cv,na.rm = T),building_id]

bond_dat[,man_cv_7 := sd(high_cv,na.rm = T),manager_id]
bond_dat[,man_cv_8 := sd(low_cv,na.rm = T),manager_id]
bond_dat[,man_cv_9 := sd(medium_cv,na.rm = T),manager_id]

bond_dat[,man_cv_10 := sd(high_cv,na.rm = T),building_id]
bond_dat[,man_cv_11 := sd(low_cv,na.rm = T),building_id]
bond_dat[,man_cv_12 := sd(medium_cv,na.rm = T),building_id]

bond_dat[,c('high_cv','low_cv','medium_cv') := NULL]

train_bond <- bond_dat[1:nrow(train)]
test_bond <- bond_dat[(nrow(train)+1):124011]
test_bond[,interest_level := NULL]

set.seed(20009)

d_train <- xgb.DMatrix(data = as.matrix(train_bond[,-c('listing_id','interest_level'),with=F]), label=train_one$interest_level, missing = NA)
#d_val <- xgb.DMatrix(data = as.matrix(train_one[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(test_bond[,-c('listing_id')]),missing = NA)


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
              subsample = 1,
              colsample_bytree = 1
) #0.5346

cv_clf <- xgb.cv(params = param
                 ,data = d_train
                 ,nrounds = 1000
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 10
                 ,maximize = F
                 #,prediction = T
)


#watch <- list(val= d_val, train=d_train)

xgb2 <- xgb.train(data = d_train,
                  params = param,
                  #watchlist=watch
                  #early_stopping_rounds = 10
                  #maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 213,
                  print_every_n = 10
                  
)

imp <- xgb.importance(xgb2,feature_names = colnames(d_train))
xgb.plot.importance(importance_matrix = imp,top_n = 20)


sPreds <- as.data.table(t(matrix(predict(xgb2, d_test), nrow=3, ncol=nrow(d_test))))

colnames(sPreds) <- c("high","low","medium")

sPreds <- data.table(listing_id=test$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "new_feat_17april_xgboost_cvfeatures.csv") #0.59054


#some new features from ankit
train_bond[,manager_count := .N,manager_id]
test_bond[,manager_count := .N,manager_id]

train_bond[,building_count := .N, building_id]
test_bond[,building_count := .N, building_id]


#interaction between manager_id and building_id
library(gtools)

vars <- c('manager_id','building_id')

cmb <- combinations(n = length(vars), r = 2, v = vars)

for(i in 1:nrow(cmb)){
  train_bond[[paste0(cmb[i,1], cmb[i,2])]] <- paste(train_bond[[cmb[i,1]]],train_bond[[cmb[i,2]]],sep = "")
  test_bond[[paste0(cmb[i,1], cmb[i,2])]] <- paste(test_bond[[cmb[i,1]]],test_bond[[cmb[i,2]]],sep = "")
  
}

train_bond[,building_idmanager_id := as.integer(building_idmanager_id)]
test_bond[,building_idmanager_id := as.integer(building_idmanager_id)]

train_bond[,id_sum := manager_id + building_id]
test_bond[,id_sum := manager_id + building_id]

train_bond[,id_diff := manager_id - building_id]
test_bond[,id_diff := manager_id - building_id]

train_bond[,id_div := manager_id/building_id]
test_bond[,id_div := manager_id/building_id]

train_bond[,id_mul := manager_id*building_id]
test_bond[,id_mul := manager_id*building_id]


#remove correlated features
dc <- findCorrelation(x = cor(train_bond[,-c('listing_id','interest_level'),with=F],use = 'complete.obs'),cutoff = 0.8,names = T)
dc


#simple feat
d1 <- setdiff(colnames(train_bond),colnames(mega_train))

to_use_1 <- c('manager_level_low','manager_level_high','manager_level_medium')


d_train <- xgb.DMatrix(data = as.matrix(train_bond[,-c('listing_id','interest_level',dc),with=F]), label=target, missing = NA)
#d_val <- xgb.DMatrix(data = as.matrix(train_one[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(test_bond[,-c('listing_id')]),missing = NA)


param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = 0.1,
              gamma = 0,
              max_depth = 6,
              min_child_weight = 1,
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
                 #,prediction = T
)


xgb2 <- xgb.train(data = d_train,
                  params = param,
                  #watchlist=watch
                  #early_stopping_rounds = 10
                  #maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 50,
                  print_every_n = 10
                  
)

imp <- xgb.importance(xgb2,feature_names = colnames(d_train))
xgb.plot.importance(importance_matrix = imp,top_n = 20)


#combinining predictions from other models
load("/home/manish/Desktop/Assignment/assignment1/stacking_data.RData")

train_bond <- merge(train_bond,mega_train[,-c('price','interest_level'),with=F],by = "listing_id",all.x = T)
test_bond <- merge(test_bond,mega_test[,-c('rf_high','rf_low','rf_medium'),with=F],by='listing_id',all.x = T)

train_bond[,c('rf_high','rf_low','rf_medium') := NULL]

head(train_bond)

train_bond[is.na(train_bond)] <- -1
test_bond[is.na(test_bond)] <- -1

# ert1 <- extraTrees(x = as.matrix(train_bond[,-c('listing_id','interest_level'),with=F])
#                    ,y = as.factor(train_bond$interest_level)
#                    ,ntree = 1000
#                    ,mtry = 6
#                    ,nodesize = 5
#                    ,numRandomCuts = 6
#                    ,numThreads = 2)
# 
# ert_pred <- predict(ert1, as.matrix(test_bond[,-c('listing_id'),with=F]))

# glmn <- cv.glmnet(x = as.matrix(train_bond[,-c('listing_id','interest_level'),with=F])
#                   ,y = train_bond$interest_level
#                   ,nfolds = 5L
#                   ,family='multinomial'
#                   ,alpha=0.8)
# glmn_train <- glmnet(x = as.matrix(train_bond[,-c('listing_id','interest_level'),with=F])
#                      ,y = train_bond$interest_level
#                      ,family = 'multinomial'
#                      ,lambda = glmn$lambda.min
#                      )





#weighted glmnet + xgbost prediction





































