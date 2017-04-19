path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

library(lubridate)
library(dplyr)
library(jsonlite)
library(caret)
library(purrr)
library(xgboost)
library(MLmetrics)
library(tidytext)
library(reshape2)
seed = 1985
set.seed(seed)

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

#add features and photos count
setDT(train)
setDT(test)
train[,feature_count := unlist(lapply(features, function(x) length(x)))]
test[,feature_count := unlist(lapply(features, function(x) length(x)))]

train[,photo_count := unlist(lapply(photos, function(x) length(x)))]
test[,photo_count := unlist(lapply(photos, function(x) length(x)))]

#replace empty list with Nofeat
train[,features := ifelse(map(features, is_empty),"Nofeat",features)]
test[,features := ifelse(map(features, is_empty),"Nofeat",features)]

# Add dummy interest level ------------------------------------------------
test[,interest_level := 'none']

# Combine train and test --------------------------------------------------
train_test <- rbind(train,test)

# Features to use ---------------------------------------------------------

feat <- c("bathrooms","bedrooms","building_id", "created","latitude", "description",
          "listing_id","longitude","manager_id", "price", "features",
          "display_address", "street_address","feature_count","photo_count", "interest_level")

train_test <- train_test[,names(train_test) %in% feat,with=F]


# Process Word Features ---------------------------------------------------

word_remove = c('allowed', 'building','center', 'space','2','2br','bldg','24',
                '3br','1','ft','3','7','1br','hour','bedrooms','bedroom','true',
                'stop','size','blk','4br','4','sq','0862','1.5','373','16','3rd','block',
                'st','01','bathrooms')


# create sparse matrix for word features ----------------------------------

word_sparse <- train_test[,names(train_test) %in% c("features","listing_id"),with=F]


# create word features ----------------------------------------------------

new_word_sparse <- word_sparse%>%
  #filter(map(features, is_empty) != TRUE) %>% #this step is useless
  tidyr::unnest(features)%>% #this unlists the list into vector
  unnest_tokens(word, features)

#my_test <- data.table(listing_id = rep(unlist(train$listing_id), lapply(train$features, length)), features = unlist(train$features))


#remove stop words and other words
new_word_sparse <- new_word_sparse[!(new_word_sparse$word %in% stop_words$word),]
new_word_sparse <- new_word_sparse[!(new_word_sparse$word %in% word_remove),]

# get most common features
# top_word <- new_word_sparse%>%
#   count(word, sort=TRUE)

#top 25 words
top_word <- as.character(as.data.frame(sort(table(new_word_sparse$word), decreasing = T)[1:30])$Var1)

new_word_sparse <- new_word_sparse[new_word_sparse$word %in% top_word,]
new_word_sparse$word <- as.factor(new_word_sparse$word)

new_word_sparse <- dcast(new_word_sparse, listing_id ~ word, length, value.var = "word")

#merge word features into main data frame

train_test <- merge(train_test, new_word_sparse, by = "listing_id", sort = FALSE, all.x = TRUE)



# Non Word Features -------------------------------------------------------
setDT(train_test)

train_test[,building_id := as.integer(factor(building_id))]
train_test[,manager_id := as.integer(factor(manager_id))]

train_test[,display_address := as.integer(factor(display_address))]
train_test[,street_address := as.integer(factor(street_address))]



# Convert Date ------------------------------------------------------------

train_test[,created := ymd_hms(created)]
train_test[,month := month(created)]
train_test[,day := day(created)]
train_test[,hour := hour(created)]
train_test[,created := NULL]


# Length of description in words ------------------------------------------

train_test[,description_len := sapply(strsplit(description,"\\s+"), length)]
train_test[,description := NULL]


# Price to bedroom Ratio --------------------------------------------------

train_test[,bed_price := price/bedrooms]
train_test[which(is.infinite(bed_price))]$bed_price <- train_test[which(is.infinite(bed_price))]$price


# Add sum of rooms and price per room -------------------------------------

train_test[,room_sum := bathrooms + bedrooms]
train_test[,room_diff := bedrooms - bathrooms]
train_test[,room_price := price/room_sum]
train_test[,bed_ratio := bedrooms/room_sum]

train_test[which(is.infinite(room_price))]$room_price <- train_test[which(is.infinite(room_price))]$price



# Log transform features --------------------------------------------------
library(ggplot2)
ggplot(train_test,aes(photo_count))+geom_histogram(color="black",fill="lightblue",bins = 50)
ggplot(train_test,aes(description_len))+geom_histogram(color="black",fill="lightblue",bins = 50)
plotDX <- function(name) {
    ggplot(train_test,aes(name))+geom_bar(color='black',fill='lightblue')
  readline()

}
lapply(colnames(train_test)[sapply(train_test,is.numeric)], plotDX)

train_test[,photo_count := log(photo_count + 1)]
train_test[,feature_count := log(feature_count + 1)]

train_test[, price := log(price + 1)]

train_test[, room_price := log(room_price + 1)]
train_test[, bed_price := log(bed_price + 1)]



# Add distance similarity feature -----------------------------------------

install.packages("RecordLinkage")
library(RecordLinkage)

vec.addressimilarity <- levenshteinSim(tolower(train_test$street_address), tolower(train_test$display_address))

train_test <- cbind(train_test, vec.addressimilarity)
train_test[,vec.addressimilarity := round(vec.addressimilarity,5)]
#train_test[,vec.addressimilarity := ifelse(vec.addressimilarity >= 0.5,1,0)]



# Split train and test ----------------------------------------------------

train <- train_test[listing_id %in% train_id]
test <- train_test[listing_id %in% test_id]


# Convert labels to integers ----------------------------------------------

train[,interest_level := as.integer(factor(interest_level))]
y <- train$interest_level
y <- y - 1
train[,interest_level := NULL]
test[,interest_level := NULL]


#Parameters for XGB
xgb_params <- list(
  colsample_bytree = 0.7,
  subsample = 0.7,
  eta = 0.1,
  objective = "multi:softprob",
  max_depth = 4,
  min_child_weight = 1,
  eval_metric = "mlogloss",
  num_class = 3,
  seed = seed
  
)


#convert to xgbmatrix
dtest <- xgb.DMatrix(data.matrix(test))

#create folds
kfolds <- 10
folds <- createFolds(y = y, k = kfolds, list = T, returnTrain = F)
fold <- as.numeric(unlist(folds[1]))


#Train Set
x_train <- train[-fold] #training set
x_val <- train[fold] #out of fold validation

y_train <- y[-fold]
y_val <- y[fold]


#Convert to XGBMatrix
dtrain <- xgb.DMatrix(as.matrix(x_train), label = y_train)
dval <- xgb.DMatrix(as.matrix(x_val), label = y_val)

#perform training
gbdt <- xgb.train(params = xgb_params
                  ,data = dtrain
                  ,nrounds = 475
                  ,watchlist = list(train = dtrain, val=dval)
                  ,print_every_n = 25
                  ,early_stopping_rounds = 50)

allpredictions <- as.data.table(matrix(predict(gbdt, dtest),nrow = dim(test), byrow=T))
##Generate Submission
allpredictions = cbind (allpredictions, test$listing_id)
names(allpredictions)<-c("high","low","medium","listing_id")
fwrite(allpredictions,"starter_xgb_kaggle.csv") #0.5582

imp <- xgb.importance(names(train), model = gbdt)
xgb.ggplot.importance(imp)

#build another model on top 17 features
imp$Feature[1:17]


dtrain_17 <- xgb.DMatrix(as.matrix(x_train[,imp$Feature[1:17],with=F]), label = y_train)
dval_17 <- xgb.DMatrix(as.matrix(x_val[,imp$Feature[1:17],with=F]), label = y_val)

#perform training
gbdt <- xgb.train(params = xgb_params
                  ,data = dtrain_17
                  ,nrounds = 475
                  ,watchlist = list(train = dtrain_17, val=dval_17)
                  ,print_every_n = 25
                  ,early_stopping_rounds = 50)

allpredictions <- as.data.table(matrix(predict(gbdt, dtest),nrow = dim(test), byrow=T))
##Generate Submission
allpredictions = cbind (allpredictions, test$listing_id)
names(allpredictions)<-c("high","low","medium","listing_id")
fwrite(allpredictions,"starter_xgb_kaggle.csv")
fwrite(allpredictions,"starter_xgb_17_kaggle.csv") #1.13

































