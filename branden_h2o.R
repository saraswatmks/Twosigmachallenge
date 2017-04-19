
#new_train
#new_test

# interest_level = ts1[class > -1]
# train <- cbind(train, interest_level$class)
# setnames(train,"V2","target")

new_train[,interest_level := as.factor(train$interest_level)]

library(h2o)
h2o.init(nthreads = -1,max_mem_size = "20G")

#load data
h2otrain <- as.h2o(new_train)
h2otest <- as.h2o(new_test)

xpo <- h2o.splitFrame(data = h2otrain, ratios = 0.6,seed = 1)

h2oval <- xpo[[2]]

sd <- grep(pattern = "feature_",x = colnames(new_train),value = T)
y <- "interest_level"
x <- setdiff(colnames(new_train), c(y,sd))


#generate 2 models
h2otrain <- h2o.na_omit(h2otrain)

my_glm <- h2o.glm(x = x
                  ,y = y
                  ,training_frame = h2otrain
                  ,nfolds = 5L
                  ,keep_cross_validation_predictions = T
                  ,ignore_const_cols = T
                  ,family = 'multinomial'
                  ,seed = 1
                  ,remove_collinear_columns = T)










my_gbm <- h2o.gbm(x = x
                  ,y = y
                  ,training_frame = h2otrain
                  ,distribution = "multinomial"
                  ,fold_assignment = "Modulo"
                  ,nfolds = 5
                  ,keep_cross_validation_predictions = TRUE
                  ,seed=1
                  ,ntrees = 100)

h2o.performance(my_gbm,xval = T) #0.56


my_rf <- h2o.randomForest(x=x
                          ,y=y
                          ,training_frame = h2otrain
                          ,fold_assignment = "Modulo"
                          ,keep_cross_validation_predictions = T
                          ,nfolds = 5
                          ,ntrees = 100
                          ,ignore_const_cols = T
                          ,seed = 1
                          )

#stacking doesn't work on multinomial
ensemble <- h2o.stackedEnsemble(x = x
                                ,y = y
                                ,training_frame = h2otrain
                                ,model_id = "my_ensemble"
                                ,base_models = list(my_gbm@model_id, my_rf@model_id)
                                )

#tuned gbm
my_gbm <- h2o.gbm(x = x
                  ,y = y
                  ,training_frame = h2otrain
                  ,learn_rate = 0.05
                  ,learn_rate_annealing = 0.99
                  ,ntrees = 500
                  ,max_depth = 10
                  ,validation_frame = h2oval
                  ,ignore_const_cols = T
                  ,distribution = "multinomial")

pred_gbm <- as.data.table(t(matrix(h2o.predict(my_gbm, h2otest2), nrow=3,ncol=nrow(h2otest2))))
poli <- as.data.table(h2o.predict(my_gbm,h2otest2))

submission <- data.table(listing_id = test$listing_id, low = poli$p0, medium = poli$p1, high=poli$p2)
fwrite(submission, "brandon_h2o.csv") #0.5675




#after ensemble, build a model on different seeds








