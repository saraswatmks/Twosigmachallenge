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


#establish a baseline model
check_col <- c("bathrooms","bedrooms","building_id","created","display_address","latitude","longitude","manager_id")

X_train <- train[,check_col,with=F]
X_test <- test[,check_col,with=F]

X_train[,uniqueN(building_id)]
X_test[,uniqueN(building_id)]

X_train[,uniqueN(manager_id)]
X_test[,uniqueN(manager_id)]


building <- unique(X_test, by="building_id")
building[,ID := as.integer(as.factor(building_id))-1]
building <- building[,.(ID,building_id)]

X_train <- building[X_train, on="building_id",nomatch=NA]
X_train[,building_id := NULL]

X_test <- building[X_test, on="building_id",nomatch=NA]
X_test[,building_id := NULL]

manager <- unique(X_test, by="manager_id")
manager <- manager[,.(manager_id)]
manager[,ID := as.integer(as.factor(manager_id))-1]

X_train <- manager[X_train, on="manager_id", nomatch=NA]
X_train[,manager_id := NULL]

X_test <- manager[X_test, on="manager_id", nomatch=NA]
X_test[,manager_id := NULL]
X_test[,i.ID.1 := NULL]

setnames(X_train,c("ID","i.ID"),c("manager_ID","building_ID"))
setnames(X_test,c("ID","i.ID"),c("manager_ID","building_ID"))

address <- unique(X_test,by="display_address")
address <- address[,.(display_address)]
address[,ID := as.integer(as.factor(display_address))-1]

X_test <- address[X_test, on="display_address"]
X_train <- address[X_train, on="display_address",nomatch=NA]
setnames(X_train,c("ID"),"display_ID")
setnames(X_test,c("ID"),"display_ID")

X_train[,display_address := NULL]
X_test[,display_address := NULL]


X_train[,created := ymd_hms(created)]
# X_train[,c_year := year(created)]
# X_train[,c_year := NULL]
X_train[,c_month := month(created)]
X_train[,c_day := day(created)]
X_train[,c_hour := hour(created)]

X_test[,created := ymd_hms(created)]
# X_train[,c_year := year(created)]
# X_train[,c_year := NULL]
X_test[,c_month := month(created)]
X_test[,c_day := day(created)]
X_test[,c_hour := hour(created)]

X_train[,created := NULL]
X_test[,created := NULL]

X_train
X_test


#xgboost
train[,interest_level := as.integer(factor(interest_level))]
y <- train$interest_level
y <- y - 1


#Parameters for XGB
seed = 1
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
dtest <- xgb.DMatrix(data.matrix(X_test))

#create folds
library(caret)
kfolds <- 5
folds <- createFolds(y = y, k = kfolds, list = T, returnTrain = F)
fold <- as.numeric(unlist(folds[1]))


#Train Set
F_train <- X_train[-fold] #training set
F_val <- X_train[fold] #out of fold validation

S_train <- y[-fold]
S_val <- y[fold]


#Convert to XGBMatrix
dtrain <- xgb.DMatrix(as.matrix(F_train), label = S_train)
dval <- xgb.DMatrix(as.matrix(F_val), label = S_val)


gbdt <- xgb.train(params = xgb_params
                  ,data = dtrain
                  ,nrounds = 1000
                  ,watchlist = list(train = dtrain, val=dval)
                  ,print_every_n = 5
                  ,early_stopping_rounds = 10) #0.78 logloss



table<-data.table("Sample" = sample(40:46, 6), "Conc1" = sample(100:106,6), 
                  "id1" = as.character(sample(1:6, 6)), "Conc2" = sample(200:206,6),
                  "id2" = as.character(sample(1:6, 6))) 

key<-data.table("Name" = c("Sally", "John", "Roger", "Bob", "Kelsey", "Molly"), 
                "id1" = as.character(1:6))

table[ key, on = c("id1"), id1 := i.Name][]
table[ key, on = c(id2 = 'id1'), id2 := i.Name][]

set.seed(1)
df = data.frame( pep = replicate( 3 , paste( sample(999,3) , collapse=";") ) , pro = sample(3) , stringsAsFactors = FALSE )

setDT(df)[,list(pep = unlist(strsplit(pep, ";"))),by=pro]

dt1 <- data.table(urn = 1:10, V1=0,V2=0,V3=0)
dt2 <- data.table(urn=rep(1:10,2),classification=0)
dt2$classification <- 1:7





