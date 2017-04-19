path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

packages <- c("jsonlite","purrr","data.table","xgboost")
purrr::walk(packages, library, character.only=TRUE, warn.conflicts = F)

train <- fromJSON("train.json")
test <- fromJSON("test.json")

#Train
vars <- setdiff(names(train), c("photos", "features"))
train <- map_at(train, vars, unlist) %>% tibble::as_tibble(.)
train_id <-train$listing_id

#Test
vars <- setdiff(names(test), c("photos", "features"))
test <- map_at(test, vars, unlist) %>% tibble::as_tibble(.)
test_id <-test$listing_id

setDT(train)
setDT(test)

head(train)

library(lubridate)
train[,created := ymd_hms(created)]
test[,created := ymd_hms(created)]

train[,range(created)]
test[,range(created)]

train <- train[order(created)]
test <- test[order(created)]

train[,omonth := month(created)]
train[,oday := day(created)]
train[,ohour := hour(created)]
train[,owkday := wday(created)]
train[,oyday := yday(created)]


#explore data

library(ggplot2)
ggplot(train,aes(created))+geom_histogram(bins = 50,fill="lightblue",color="black")+
  scale_x_datetime(date_breaks = "1 weeks")

ggplot(train,aes(created,factor(interest_level)))+
  geom_bar(fill="lightblue",color="black",stat="identity")+
  scale_x_datetime(date_breaks = "1 weeks")

ggplot(train,aes(created,price))+
  geom_bar(fill="lightblue",color="black",stat="identity")+
  scale_x_datetime(date_breaks = "1 weeks")

ggplot(train, aes(omonth))+
  geom_histogram(mapping = aes(color = as.factor(interest_level)), binwidth=100)

ggplot(train,aes(x = as.factor(omonth),fill = as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(oday), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(ohour), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(owkday), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(bathrooms), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(bedrooms), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(building_id), fill=as.factor(interest_level)))+geom_bar()
ggplot(train,aes(as.factor(building_id), fill=as.factor(interest_level)))+geom_bar()

ggplot(train[order(listing_id)],aes(as.factor(listing_id), fill=as.factor(interest_level)))+geom_bar()


ggplot(train[order(oyday)],aes(as.factor(oyday), fill=as.factor(interest_level)))+geom_bar()+
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

#

#Feature Engineering
train[,feature_count := unlist(lapply(features, function(x) length(x)))]
test[,feature_count := unlist(lapply(features, function(x) length(x)))]

train[,photo_count := unlist(lapply(photos, function(x) length(x)))]
test[,photo_count := unlist(lapply(photos, function(x) length(x)))]

train[,features := ifelse(map(features, is_empty),"Nofeat",features)]
test[,features := ifelse(map(features, is_empty),"Nofeat",features)]



#price has outliers

#1 listing has 10 bathrooms, outlier it is

#listing id vs interest level

#num of words in a description, length of description


# Got from Brandon's Solution ---------------------------------------------

new_train <- ts1[filter == 0, varnames, with=F]
new_test <- ts1[filter == 2, varnames, with=F]

new_train[,interest_level := train$interest_level]
new_train[,.N,interest_level]

emp <- model.matrix(~.+0, data = new_train[,c("interest_level")])

new_train <- cbind(new_train, emp)

new_train[,interest_levelhigh := mean(interest_levelhigh),manager_id]
new_train[,interest_levellow := mean(interest_levellow),manager_id]
new_train[,interest_levelmedium := mean(interest_levelmedium),manager_id]
setnames(new_train,c("interest_levelhigh","interest_levellow","interest_levelmedium"),c("high_frac_man","low_frac_man","medium_frac_man"))

new_train[,interest_levelhigh := mean(interest_levelhigh), building_id]
new_train[,interest_levellow := mean(interest_levellow),building_id]
new_train[,interest_levelmedium := mean(interest_levelmedium),building_id]
setnames(new_train,c("interest_levelhigh","interest_levellow","interest_levelmedium"),c("high_frac_bul","low_frac_bul","medium_frac_bul"))

new_train[,created := train$created]
new_train[,created := ymd_hms(created)]
new_train[,omonth := month(created)]
new_train[,oday := day(created)]
new_train[,owday := wday(created)]
new_train[,ohour := hour(created)]

new_test[,created := test$created]
new_test[,created := ymd_hms(created)]
new_test[,omonth := month(created)]
new_test[,oday := day(created)]
new_test[,owday := wday(created)]
new_test[,ohour := hour(created)]

new_train[,interest_levelhigh_hour := mean(interest_levelhigh),ohour]
new_train[,interest_levellow_hour := mean(interest_levellow),ohour]
new_train[,interest_levelmedium_hour := mean(interest_levelmedium),ohour]

new_train[,interest_levelhigh_day := mean(interest_levelhigh),oday]
new_train[,interest_levellow_day := mean(interest_levellow),oday]
new_train[,interest_levelmedium_day := mean(interest_levelmedium),oday]

new_train[,c("interest_levelhigh","interest_levellow","interest_levelmedium") := NULL]

check <- setdiff(colnames(new_train),colnames(new_test))

class <- data.table(interest_level = c("low","medium","high"), class=c(0,1,2))
new_train <- class[new_train, on="interest_level"]

man <- new_train[,.(manager_id, high_frac_man, low_frac_man, medium_frac_man)]
man <- unique(man)
new_test <- man[new_test, on="manager_id"]

bul <- new_train[,.(building_id, high_frac_bul, low_frac_bul, medium_frac_bul)]
bul <- unique(bul)
new_test <- bul[new_test, on="building_id"]

hou <- new_train[,.(ohour,interest_levelhigh_hour,interest_levellow_hour,interest_levelmedium_hour)]
hou <- unique(hou)
new_test <- hou[new_test, on="ohour"]

ciday <- new_train[,.(oday, interest_levelhigh_day,interest_levellow_day,interest_levelmedium_day)]
ciday <- unique(ciday)
new_test <- ciday[new_test, on="oday"]

setdiff(colnames(new_train),colnames(new_test))

new_train[,interest_level := NULL]


library(caTools)
split <- sample.split(Y = new_train$class, SplitRatio = 0.6)

Y <- new_train$class
listing_id <- new_train$listing_id

new_train[,c("listing_id","class") := NULL]
new_train[,created := NULL]
new_test[,created := NULL]

d_train <- xgb.DMatrix(data = as.matrix(new_train), label=Y, missing = NA)
d_val <- xgb.DMatrix(data = as.matrix(new_train[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(new_test))


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

set.seed(201609)

watch <- list(val= d_val, train=d_train)

xgb2 <- xgb.train(data = d_train,
                  params = param,
                  watchlist=watch
                  ,early_stopping_rounds = 10
                  ,maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 2000,
                  print_every_n = 10
                  
)

xgb.plot.importance(importance_matrix = xgb.importance(feature_names = colnames(new_train), model = xgb2),top_n = 50)


sPreds <- as.data.table(t(matrix(predict(xgb2, d_test), nrow=3, ncol=nrow(d_test))))
colnames(sPreds) <- class$interest_level

sPreds <- data.table(listing_id=test$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "combine_features_xgboost.csv")

cx <- grep(pattern = "feature_", x = colnames(new_train),value = T)

# 
# [1] "bathrooms"                 "bedrooms"                  "building_id"               "display_address"          
# [5] "latitude"                  "longitude"                 "manager_id"                "price"                    
# [9] "street_address"            "photo_count"               "building_id_mean_med"      "manager_id_mean_med"      
# [13] "building_id_mean_high"     "manager_id_mean_high"      "desc_wordcount"            "pricePerBed"              
# [17] "pricePerBath"              "pricePerRoom"              "bedPerBath"                "bedBathDiff"              
# [21] "bedBathSum"                "bedsPerc"                  "high_frac_man"             "low_frac_man"             
# [25] "medium_frac_man"           "high_frac_bul"             "low_frac_bul"              "medium_frac_bul"          
# [29] "omonth"                    "oday"                      "owday"                     "ohour"                    
# [33] "interest_levelhigh_hour"   "interest_levellow_hour"    "interest_levelmedium_hour" "interest_levelhigh_day"   
# [37] "interest_levellow_day"     "interest_levelmedium_day" 


first_batch <- colnames(train_one)[1:10] #0.58829 LB, #0.434678 CV
second_batch <- c(first_batch,"desc_wordcount") #0.426269 CV #0.64885 LB
third_batch <- c(first_batch, "pricePerBed","pricePerBath","pricePerRoom","bedPerBath","bedBathDiff","bedBathSum","bedsPerc")

#continue from third batch

train_one <- new_train[,third_batch,with=F]
test_one <- new_test[,third_batch,with=F]

d_train <- xgb.DMatrix(data = as.matrix(train_one), label=Y, missing = NA)
d_val <- xgb.DMatrix(data = as.matrix(train_one[!split]), label=Y[!split], missing = NA)
d_test <- xgb.DMatrix(data = as.matrix(test_one))


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
                  watchlist=watch
                  ,early_stopping_rounds = 10
                  ,maximize = F,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 400,
                  print_every_n = 10
                  
)

sPreds <- as.data.table(t(matrix(predict(xgb2, d_test), nrow=3, ncol=nrow(d_test))))
colnames(sPreds) <- class$interest_level

sPreds <- data.table(listing_id=test$listing_id, sPreds[,list(high,medium,low)])
fwrite(sPreds, "combine_features_xgboost_11.csv")











































