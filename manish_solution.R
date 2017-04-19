path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

packages <- c("jsonlite","dplyr","purrr","data.table","xgboost","caret","stringr","quanteda","lubridate","Hmisc","Matrix")
purrr::walk(packages, library, character.only=TRUE, warn.conflicts = F)

catNWayAvgCV <- function(data, varList, y, pred0, filter, k, f, g=1, lambda=NULL, r_k, cv=NULL){
  # It is probably best to sort your dataset first by filter and then by ID (or index)
  n <- length(varList)
  varNames <- paste0("v",seq(n))
  ind <- unlist(cv, use.names=FALSE)
  oof <- NULL
  if (length(cv) > 0){
    for (i in 1:length(cv)){
      sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
      sub1 <- sub1[sub1$filt==TRUE,]
      sub1[,filt:=NULL]
      colnames(sub1) <- c(varNames,"y","pred0")
      sub2 <- sub1[cv[[i]],]
      sub1 <- sub1[-cv[[i]],]
      sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
      tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
      set(tmp1, i=which(is.na(tmp1[,cnt])), j="cnt", value=0)
      set(tmp1, i=which(is.na(tmp1[,sumy])), j="sumy", value=0)
      if(!is.null(lambda)) tmp1[beta:=lambda] else tmp1[,beta:= 1/(g+exp((tmp1[,cnt] - k)/f))]
      tmp1[,adj_avg:=((1-beta)*avgY+beta*pred0)]
      set(tmp1, i=which(is.na(tmp1[["avgY"]])), j="avgY", value=tmp1[is.na(tmp1[["avgY"]]), pred0])
      set(tmp1, i=which(is.na(tmp1[["adj_avg"]])), j="adj_avg", value=tmp1[is.na(tmp1[["adj_avg"]]), pred0])
      set(tmp1, i=NULL, j="adj_avg", value=tmp1$adj_avg*(1+(runif(nrow(sub2))-0.5)*r_k))
      oof <- c(oof, tmp1$adj_avg)
    }
  }
  oofInd <- data.frame(ind, oof)
  oofInd <- oofInd[order(oofInd$ind),]
  sub1 <- data.table(v1=data[,varList,with=FALSE], y=data[,y,with=FALSE], pred0=data[,pred0,with=FALSE], filt=filter)
  colnames(sub1) <- c(varNames,"y","pred0","filt")
  sub2 <- sub1[sub1$filt==F,]
  sub1 <- sub1[sub1$filt==T,]
  sum1 <- sub1[,list(sumy=sum(y), avgY=mean(y), cnt=length(y)), by=varNames]
  tmp1 <- merge(sub2, sum1, by = varNames, all.x=TRUE, sort=FALSE)
  tmp1$cnt[is.na(tmp1$cnt)] <- 0
  tmp1$sumy[is.na(tmp1$sumy)] <- 0
  if(!is.null(lambda)) tmp1$beta <- lambda else tmp1$beta <- 1/(g+exp((tmp1$cnt - k)/f))
  tmp1$adj_avg <- (1-tmp1$beta)*tmp1$avgY + tmp1$beta*tmp1$pred0
  tmp1$avgY[is.na(tmp1$avgY)] <- tmp1$pred0[is.na(tmp1$avgY)]
  tmp1$adj_avg[is.na(tmp1$adj_avg)] <- tmp1$pred0[is.na(tmp1$adj_avg)]
  # Combine train and test into one vector
  return(c(oofInd$oof, tmp1$adj_avg))
}


#load training data
print("load training data")
t1 <- fromJSON("train.json")
t1_feats <- data.table(listing_id = rep(unlist(t1$listing_id), lapply(t1$features, length)), features = unlist(t1$features))
head(t1_feats)
t1_photos <- data.table(listing_id = rep(unlist(t1$listing_id), lapply(t1$photos, length)), photos= unlist(t1$photos))
head(t1_photos)

vars <- setdiff(names(t1), c("photos", "features"))
t1<- map_at(t1, vars, unlist) %>% as.data.table(.)

head(t1)
t1[,filter := 0]


#create 5 fold CV
library(caret)
set.seed(321)
cvFoldsList <- createFolds(t1$interest_level, k=5, list = T, returnTrain = F)

#convert class to integers
class <- data.table(interest_level = c("low","medium","high"), class=c(0,1,2))
t1 <- merge(t1,class, by='interest_level', all.x = TRUE, sort = F)
head(t1)

#load test data
print("loading test data")
s1 <- fromJSON("test.json")

s1_feats <- data.table(listing_id = rep(unlist(s1$listing_id), lapply(s1$features, length)), features = unlist(s1$features))
s1_photos <- data.table(listing_id = rep(unlist(s1$listing_id), lapply(s1$photos, length)), photos = unlist(s1$photos))

vars <- setdiff(names(s1), c("photos", "features"))
s1<- map_at(s1, vars, unlist) %>% as.data.table(.)
setDT(s1)

s1[,c("interest_level","class","filter") := list("-1",-1,2)]

ts1 <- rbind(t1,s1)

rm(t1,s1);gc()

ts1_feats <- rbind(t1_feats, s1_feats)
rm(t1_feats,s1_feats)

ts1_photos <- rbind(t1_photos,s1_photos)
rm(t1_photos,s1_photos)

ts1[,":="(created=as.POSIXct(created)
          ,dummy="A"
          ,low=as.integer(interest_level=="low")
          ,medium=as.integer(interest_level=="medium")
          ,high=as.integer(interest_level=="high")
          ,display_address=trimws(tolower(display_address))
          ,street_address=trimws(tolower(street_address)))]

ts1[, ":="(pred0_low=sum(interest_level=="low")/sum(filter==0),
           pred0_medium=sum(interest_level=="medium")/sum(filter==0),
           pred0_high=sum(interest_level=="high")/sum(filter==0))]


#merge feature column
ts1_feats[,features := gsub(" ","_", paste0("feature_",trimws(char_tolower(features))))]

feats_summ <- ts1_feats[,.N,features]

ts1_feats_cast <- dcast(ts1_feats[!features %in% feats_summ[N < 10, features]], listing_id ~ features, fun.aggregate = function(x) as.integer(length(x) > 0), value.var = "features")

ts1 <- merge(ts1, ts1_feats_cast, by="listing_id", all.x = T, sort = F)
rm(ts1_feats_cast);
gc()

#photo counts
ts1_photos_summ <- ts1_photos[,.(photo_count = .N), by=listing_id]
ts1 <- merge(ts1, ts1_photos_summ, by = "listing_id", all.x = T, sort = F)

#ts1[,photo_count_cat := ifelse(photo_count >=16,1,0)]

rm(ts1_photos,ts1_photos_summ)
gc()

#convert building_id and manager_id
build_count <- ts1[,.(.N), building_id]
manag_count <- ts1[,.(.N),manager_id]
add_count <- ts1[,.(.N),display_address]

set(ts1, i = which(ts1[['building_id']] %in% build_count[N==1, building_id]), j="building_id", value="-1")
set(ts1, i = which(ts1[["manager_id"]] %in% manag_count[N == 1, manager_id]), j="manager_id", value = "-1")
set(ts1, i = which(ts1[["display_address"]] %in% add_count[N == 1, display_address]), j="display_address", value = "-1")

#mean target encoding
print("target encoding")
highCard <- c('building_id','manager_id')

for(col in 1:length(highCard)){
  ts1[,paste0(highCard[col],"_mean_med") := catNWayAvgCV(ts1, varList = c("dummy", highCard[col]), y = "medium",pred0 = "pred0_medium",filter=ts1$filter == 0, k = 5, f=1, r_k = 0.01, cv=cvFoldsList)]
}

for(col in 1:length(highCard)){
  ts1[,paste0(highCard[col], "_mean_high") := catNWayAvgCV(ts1, varList = c("dummy", highCard[col]), y="high",pred0 = "pred0_high",filter = ts1$filter == 0, k = 5,f = 1, r_k = 0.01, cv = cvFoldsList)]
}

#create some more features
print("creating some more features")
ts1[,":="(building_id=as.integer(as.factor(building_id))
          ,display_address=as.integer(as.factor(display_address))
          ,manager_id=as.integer(as.factor(manager_id))
          ,street_address=as.integer(as.factor(street_address))
          ,desc_wordcount=str_count(description)
          ,pricePerBed=ifelse(!is.finite(price/bedrooms),-1, price/bedrooms)
          ,pricePerBath=ifelse(!is.finite(price/bathrooms),-1, price/bathrooms)
          ,pricePerRoom=ifelse(!is.finite(price/(bedrooms+bathrooms)),-1, price/(bedrooms+bathrooms))
          ,bedPerBath=ifelse(!is.finite(bedrooms/bathrooms), -1, price/bathrooms)
          ,bedBathDiff=bedrooms-bathrooms
          ,bedBathSum=bedrooms+bathrooms
          ,bedsPerc=ifelse(!is.finite(bedrooms/(bedrooms+bathrooms)), -1, bedrooms/(bedrooms+bathrooms)))
    ]

print("fill in missing values")
for(col in 1:ncol(ts1))
  set(ts1, i = which(is.na(ts1[[col]])), j=col, value = -1)

# print("add variable from starter script")
# col_traintest <- train_test[,setdiff(colnames(train_test),colnames(train)),with=F]
# col_traintest[,listing_id := train_test$listing_id]
# ts1 <- merge(ts1, col_traintest, by="listing_id", all.x = T, sort = F)

print("get variable names")
varnames <- setdiff(colnames(ts1), c("photos","pred0_high", "pred0_low","pred0_medium","description", "features","interest_level","dummy","filter", "created", "class", "low","medium","high","street","listing_id"))



new_train <- ts1[filter == 0, varnames, with=F]
new_test <- ts1[filter == 2, varnames, with=F]

#continue to H2o from here

print("converting data to sparse format")
t1_sparse <- Matrix(as.matrix(ts1[filter==0, varnames, with=FALSE]), sparse=TRUE)
s1_sparse <- Matrix(as.matrix(ts1[filter==2, varnames, with=FALSE]), sparse=TRUE)

listing_id_test <- ts1[filter %in% c(2), listing_id]
labels <- ts1[filter %in% c(0), class]

print("converting data into xgb format")
dtrain <- xgb.DMatrix(data=t1_sparse, label=labels)
dval <- xgb.DMatrix(data = t1_sparse[cvFoldsList$Fold1,], label= labels[cvFoldsList$Fold1], missing = NA)
dtest <- xgb.DMatrix(data=s1_sparse)

param <- list(booster="gbtree",
              objective="multi:softprob",
              eval_metric="mlogloss",
              #nthread=13,
              num_class=3,
              eta = .02,
              gamma = 1,
              max_depth = 4,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .5
)

set.seed(20170408)

cv_clf <- xgb.cv(params = param
                 ,data = dtrain
                 ,nrounds = 500
                 ,nfold = 5L
                 ,metrics = "mlogloss"
                 ,stratified = T
                 ,print_every_n = 10
                 ,early_stopping_rounds = 10
                 ,maximize = F
                 ,prediction = T
)
cv_pred_xgboost <- as.data.table(cv_clf$pred)

watch <- list(dtrain=dtrain)

xgb2 <- xgb.train(data = dtrain,
                  params = param,
                  # watchlist=watch,
                  # nrounds = xgb2cv$best_ntreelimit
                  nrounds = 2710,
                  print_every_n = 10
)

feat_imp <- xgb.importance(feature_names = colnames(new_train), model = xgb2)
xgb.plot.importance(importance_matrix = feat_imp,top_n = 30)

sPreds <- as.data.table(t(matrix(predict(xgb2, dtest), nrow=3, ncol=nrow(dtest))))
colnames(sPreds) <- class$interest_level

test_pred_xgboost <- sPreds

fwrite(data.table(listing_id=listing_id_test, sPreds[,list(high,medium,low)]), "brandonsubmission.csv") #0.54002
fwrite(data.table(listing_id=listing_id_test, sPreds[,list(high,medium,low)]), "brandon+startersubmission.csv") #0.54346


#xgboost
#bagging different seeds

all_pred = {}
for (seed in c(201701,201601,201501,201401)){
  for (subsample in c(0.2,0.6,0.9)){
    param <- list(
      booster="gbtree",
      objective="multi:softprob",
      #eval_metric="mlogloss",
      #nthread=13,
      num_class=3,
      eta = .025,
      gamma = 1,
      max_depth = 8,
      min_child_weight = 1,
      subsample = .7,
      colsample_bytree = .7
    )
    
    print(paste("printing",seed))
    set.seed(seed)
    
    clf2 <- xgb.train(   params              = param, 
                         data                = dtrain,
                         nrounds             = 1000, 
                         #verbose             = 0,
                         early_stop_round    = 50,
                         watchlist           = list(val = dval, train = dtrain),
                         maximize            = FALSE,
                        eval_metric       = "mlogloss",
                        print_every_n = 10
                         
    )
    
    sPreds <- t(matrix(predict(clf2, dtest), nrow=3, ncol=nrow(dtest)))
    colnames(sPreds) <- c("low","medium","high")
    print (head(all_pred))
    all_pred <- cbind(all_pred, sPreds)
  }
}

all_pred <- as.data.table(all_pred)

colnames(all_pred)[seq(1,36,3)]

all_pred[,row_final := rowMeans(.SD), .SDcols = colnames(all_pred)[seq(1,36,3)]]
all_pred[,medium_final := rowMeans(.SD), .SDcols = colnames(all_pred)[seq(2,36,3)]]
all_pred[,high_final := rowMeans(.SD), .SDcols = colnames(all_pred)[seq(3,36,3)]]

all_sub <- data.table(listing_id = listing_id_test, low = all_pred$row_final, medium = all_pred$medium_final, high = all_pred$high_final)
 
fwrite(all_sub,"bag_xgboost.csv") #0.54692




