path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

library(data.table)
library(lubridate)
library(text2vec)
packages <- c("jsonlite")

library(purrr)
walk(packages,library,character.only=TRUE,warn.conflicts=TRUE)

traind <- fromJSON("train.json")
test <- fromJSON("test.json")

#unlist
vars <- setdiff(names(traind),c("photos","features"))
train <- map_at(traind, vars, unlist) %>% as.data.table()
test <- map_at(test,vars,unlist) %>% as.data.table()

x_train <- train[,.(listing_id,description)]
x_test <- test[,.(listing_id,description)]

#preprocessing
prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(x_train$description, 
                   preprocessor = prep_fun, 
                   tokenizer = tok_fun, 
                   ids = x_train$listing_id, 
                   progressbar = FALSE)

vocab <- create_vocabulary(it_train)
vocab <- create_vocabulary(it_train, stopwords = c(stop_words$word,"br"))
#or
pruned_vocab <- prune_vocabulary(vocab, term_count_min = 5, doc_proportion_min = 0.1)

vectorizer <- vocab_vectorizer(pruned_vocab)

dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)

#create a data table - 1 gram
myt <- as.data.table(as.matrix(dtm_train))
myt[,listing_id := train$listing_id]

#create a data table - 2gram
vocab <- create_vocabulary(it_train, ngram = c(1L,2L),stopwords = c(stop_words$word,"br"))
vocab <- prune_vocabulary(vocab, term_count_min = 5,doc_proportion_min = 0.05)

bigram_vec <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, bigram_vec)
str(dtm_train)

myt_twogram <- as.data.table(as.matrix(dtm_train))


#feature hashng
h_vectorizer <- hash_vectorizer(hash_size = 2^14, ngram = c(1L,2L)) #hashsize is number of columns

dh_train <- create_dtm(it_train, h_vectorizer)
str(dh_train)

hash_train <- as.data.table(as.matrix(dh_train))


#TF-IDF
vocab <- create_vocabulary(it_train)
vectorizer <- vocab_vectorizer(vocab)
dtm_train <- create_dtm(it_train, vectorizer)

tfidf <- TfIdf$new()
dtm_train_tfidf <- fit_transform(dtm_train, tfidf)
class(dtm_train_tfidf)

myt_tfidf <- as.data.table(as.matrix(dtm_train))



