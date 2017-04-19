path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

library(data.table)
library(lubridate)
library(purrr)
packages <- c("jsonlite")

library(purrr)
walk(packages,library,character.only=TRUE,warn.conflicts=TRUE)

traind <- fromJSON("train.json")
test <- fromJSON("test.json")

#unlist
vars <- setdiff(names(traind),c("photos","features"))
train <- map_at(traind, vars, unlist) %>% as.data.table()
test <- map_at(test,vars,unlist) %>% as.data.table()


#data explortion
head(train)
train[,.N/nrow(train),interest_level]


#practice with features
train[,features := ifelse(map(features, is_empty), "Nofeat", features)]
test[,features := ifelse(map(features, is_empty), "Nofeat", features)]

#combine train and test
test[,interest_level := 'none']
train_test <- rbind(train,test)



# Features ----------------------------------------------------------------
library(tidytext)
library(dplyr)

word_remove = c('allowed', 'building','center', 'space','2','2br','bldg','24',
                '3br','1','ft','3','7','1br','hour','bedrooms','bedroom','true',
                'stop','size','blk','4br','4','sq','0862','1.5','373','16','3rd','block',
                'st','01','bathrooms','in','br')

word_sparse <- train_test[,.(listing_id,features)]

new_word_sparse <- word_sparse%>%
  #filter(map(features, is_empty) != TRUE)%>%
  tidyr::unnest(features)%>%
  unnest_tokens(word, features)

#remove stop words
new_word_sparse <- new_word_sparse[!(new_word_sparse$word %in% stop_words$word),]
new_word_sparse <- new_word_sparse[!(new_word_sparse$word %in% word_remove),]

top_words <- new_word_sparse%>%
  count(word, sort=TRUE)

top_words <- top_words$word[1:30]

new_word_sparse <- new_word_sparse[new_word_sparse$word %in% top_words,]
new_word_sparse$word <- as.factor(new_word_sparse$word)

make_new_word_sparse <- dcast(new_word_sparse, listing_id ~ word, length, value.var = "word")


#now fuse these variables into main data by listing_id

train_test <- merge(train_test, make_new_word_sparse, by = "listing_id",all.x = TRUE,sort = F)
rm(make_new_word_sparse,new_word_sparse)


#Description
dt_train <- itoken(train_test$description
                   ,preprocessor = tolower
                   ,tokenizer = word_tokenizer
                   ,progressbar = FALSE
                   ,ids = train_test$listing_id)

vocab <- create_vocabulary(dt_train,stopwords = c(stop_words$word,word_remove))

pruned_vocab <- prune_vocabulary(vocab,doc_proportion_min = 0.10)

vectorizer <- vocab_vectorizer(pruned_vocab)

dtm_train <- create_dtm(dt_train, vectorizer)
dim(dtm_train)

mydesc <- as.data.table(as.matrix(dtm_train))
mydesc[,listing_id := train_test$listing_id]

train_test <- merge(train_test, mydesc, by = "listing_id",all.x = T,sort = F)
rm(mydesc)

train_test[,feature]

#create more features
train_test[,feature_count := unlist(lapply(features.x, function(x) length(x)))]
train_test[,description_len := lapply(strsplit(description,split = "\\s+"), length)]
train_test[,description_len := unlist(description_len)]
train_test[,photos_count := unlist(lapply(photos, function(x) length(x)))]

train_test <- train_test[order(listing_id)]
train_test[,seq_listing := .I, listing_id]

train_test[,building_id := as.integer(as.factor(building_id))]
train_test[,display_address := as.integer(as.factor(display_address))]
train_test[,manager_id := as.integer(as.factor(manager_id))]
train_test[,street_address := as.integer(as.factor(street_address))]

train_test[,created := ymd_hms(created)]

train_test <- train_test[order(created)]
train_test[,seq_time := .I, created]

train_test[,bed_price := price/bedrooms]
train_test[which(is.infinite(bed_price))]$bed_price <- train_test[which(is.infinite(bed_price))]$price

train_test[,room_sum := bathrooms + bedrooms]
train_test[,room_diff := bedrooms - bathrooms]
train_test[,room_price := price/room_sum]
train_test[,bed_ratio := bedrooms/room_sum]

train_test[which(is.infinite(room_price))]$room_price <- train_test[which(is.infinite(room_price))]$price

train_test[,photo_count := log(photo_count + 1)]
train_test[,feature_count := log(feature_count + 1)]
train_test[, price := log(price + 1)]
train_test[, room_price := log(room_price + 1)]
train_test[, bed_price := log(bed_price + 1)]

train_test[,month := month(created)]
train_test[,day := day(created)]
train_test[,hour := hour(created)]
train_test[,created := NULL]

#remove variables
train_test[,c("description","photos","features.x") := NULL]
train_test[,features.y := NULL]

#break the data
x_train <- train_test[1:nrow(train)]
x_test <- train_test[-(1:nrow(train))]

x_test[,interest_level := NULL]

rm(word_sparse,train_test)

#modeling
# Convert labels to integers ----------------------------------------------
library(xgboost)

x_train[,interest_level := as.integer(factor(interest_level))]
y <- train$interest_level
y <- as.integer(as.factor(y)) - 1
x_train[,interest_level := NULL]


#Parameters for XGB
seed=101
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
dtest <- xgb.DMatrix(data.matrix(x_test))

#create folds
kfolds <- 5
folds <- createFolds(y = y, k = kfolds, list = T, returnTrain = F)
str(folds)
fold <- as.numeric(unlist(folds[1]))


#Train Set
X_train <- x_train[-fold] #training set
X_val <- x_train[fold] #out of fold validation

Y_train <- y[-fold]
Y_val <- y[fold]


#Convert to XGBMatrix
dtrain <- xgb.DMatrix(as.matrix(X_train), label = Y_train)
dval <- xgb.DMatrix(as.matrix(Y_val), label = Y_val)

#perform training
gbdt <- xgb.train(params = xgb_params
                  ,data = dtrain
                  ,nrounds = 500
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


#######################XGBOost isn't working###################################
#########################using DL now#############################

x_train[,interest_level := train$interest_level]
x_train[,interest_level := as.factor(interest_level)]

library(h2o)
h2o.init(nthreads = -1, max_mem_size = "16G")

h2o_xtrain <- as.h2o(x_train)
h2o_xtest <- as.h2o(x_test)

y <- "interest_level"
x <- setdiff(colnames(x_train),c("listing_id",y))


dl_model <- h2o.deeplearning(x = x
                             ,y = y
                              ,training_frame = h2o_xtrain
                             ,nfolds = 10L
                             ,ignore_const_cols = T
                             ,standardize = T
                             ,activation = "Rectifier"
                             ,hidden = c(30,30)
                             ,epochs = 500
                             ,seed = 1
                             )
h2o.performance(dl_model)


















