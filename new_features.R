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

join[,display_address := as.integer(as.factor(display_address))]
join[,street_address := as.integer(as.factor(street_address))]

join[,latitude := NULL]

train_one <- join[1:nrow(train)]
test_one <- join[(nrow(train)+1):124011]

target <- as.integer(as.factor(train_one$interest_level))-1

############round one features########################3
train_one <- merge(train_one,manager_train[,.(listing_id,manager_level_low)],by = "listing_id",all.x = T)
test_one <- merge(test_one,manager_test[,.(listing_id,manager_level_low)],by = "listing_id",all.x = T)

train_one[,manager_level_low := NULL]
test_one[,manager_level_low := NULL]

train_one <- merge(train_one,manager_train[,.(listing_id,manager_level_high)],by = "listing_id",all.x = T)
test_one <- merge(test_one,manager_test[,.(listing_id,manager_level_high)],by = "listing_id",all.x = T)

train_one[,manager_level_high := NULL]
test_one[,manager_level_high := NULL]

train_one <- merge(train_one,manager_train[,.(listing_id,manager_level_medium)],by = "listing_id",all.x = T)
test_one <- merge(test_one,manager_test[,.(listing_id,manager_level_medium)],by = "listing_id",all.x = T)

train_one[,manager_level_medium := NULL]
test_one[,manager_level_medium := NULL]

train_one[,manager_count := .N, manager_id]
test_one[,manager_count := .N, manager_id]

train_one[,building_count := .N, building_id]
test_one[,building_count := .N, building_id]

train_one[,c('manager_count','building_count') := NULL]
test_one[,c('manager_count','building_count')]

train_one[,x1 := (bathrooms - mean(bathrooms))/sd(bathrooms), manager_id]
train_one[,x2 := (bedrooms - mean(bedrooms))/sd(bedrooms),manager_id]
train_one[,x3 := (price - mean(price))/sd(price),manager_id]

test_one[,x1 := (bathrooms - mean(bathrooms))/sd(bathrooms), manager_id]
test_one[,x2 := (bedrooms - mean(bedrooms))/sd(bedrooms),manager_id]
test_one[,x3 := (price - mean(price))/sd(price),manager_id]

train_one[,x2 := NULL]
test_one[,x2 := NULL]

train_one[,x4 := (bathrooms - mean(bathrooms))/sd(bathrooms), building_id]
train_one[,x4 := NULL]

train_one[,x5 := (bedrooms - mean(bedrooms))/sd(bedrooms), building_id]
train_one[,x5 := NULL]

train_one[,x6 := (price - mean(price))/sd(price), building_id]
train_one[,x6 := NULL]

#########round 2 features##################33
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


################skip round 2 features#################
######################## round 3###########################
setDT(train)
train[,features := ifelse(map(features, is_empty),"Nofeat",features)]

feat_train <- data.table(listing_id = rep(unlist(train$listing_id), lapply(train$features, length)), features = unlist(train$features), target = rep(unlist(train$interest_level), lapply(train$features, length)))
feat_train$target[1:10]
feat_train[,target := as.integer(as.factor(target))-1] #medium 2, low 1, high 0

feat_train[,feat_ave := mean(target), features]
feat_train[,summary(feat_ave)]
feat_train

feat_train[,feat_super_ave := mean(feat_ave), listing_id]
un_Feat <- unique(feat_train[,.(listing_id,feat_super_ave)])


train_one <- merge(train_one,un_Feat,by="listing_id",all.x = T)


dtrain <- xgb.DMatrix(data = as.matrix(train_one[,-c('listing_id','interest_level'),with=F]), label=target)
dtest <- xgb.DMatrix(data = as.matrix(test_one[,-c('listing_id','interest_level'),with=F]))

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


#0.571469
#0.790604+0.004221
#0.790507



#now starting with round 4 features#################












