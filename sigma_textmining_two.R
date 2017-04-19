path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

library(data.table)
library(lubridate)
packages <- c("jsonlite")

library(purrr)
walk(packages,library,character.only=TRUE,warn.conflicts=TRUE)

traind <- fromJSON("train.json")
test <- fromJSON("test.json")

#unlist
vars <- setdiff(names(traind),c("photos","features"))
train <- map_at(traind, vars, unlist) %>% as.data.table()
test <- map_at(test,vars,unlist) %>% as.data.table()

head(train)

colSums(is.na(train))
colSums(is.na(test))

#Lets do Text Mining 
train.text <- train[,.(description, features,interest_level)]
test.text <- test[,.(description, features)]

try <- train.text[,.(features)]
unlist(try$features)

try <- head(try)
x <- do.call(rbind, try$features)
colnames(x) <- LETTERS[1:ncol(x)]
x <- data.frame(x)

try[,count := lapply(features, function(x) length(x))]
try[,count := unlist(count)]
try[,max(count)]
try[,.N,count][order(N)]


#split a list intor rows
for(i in 1:39){
  S <- paste(paste("try[,A",i,sep = "_"),":= lapply(features, `[`,",i,")]")
  eval(parse(text = S))
}

colSums(is.na(try))


#lets remove features from train.text and derive features from description
train.text <- train[,.(description, interest_level)]
test.text <- test.text[,.(description)]

#Lets do text mining
library(tidytext)
library(dplyr)
mytext <- train.text$description

word_count <- train.text%>%unnest_tokens(word, description)
removed_words <- word_count %>% anti_join(stop_words)

removed_words%>%count(word, sort=T)

get_sentiments("nrc")

nrcjoy <- get_sentiments("nrc")%>%
  filter(sentiment == "joy")

#sentiment = joy
removed_words%>%
  inner_join(nrcjoy)%>%
  count(word, sort=TRUE)

library(wordcloud)

removed_words%>%
  count(word)%>%
  with(wordcloud(word, n, max.words = 100))

###################Text Mining with TM##############################
library(tm)
text_corpus <- Corpus(VectorSource(train$description))
print(text_corpus)
inspect(text_corpus[1:4])

#the corpus is a list object in R of type CORPUS
print(as.character(text_corpus[[1]]))
print(lapply(text_corpus[1:2], as.character))

#let's clean the data
dropword <- "br"

#remove br
text_corpus <- tm_map(text_corpus,removeWords,dropword)
print(as.character(text_corpus[[1]]))
#tolower
text_corpus <- tm_map(text_corpus, tolower)
print(as.character(text_corpus[[1]]))
#remove punctuation
text_corpus <- tm_map(text_corpus, removePunctuation)
print(as.character(text_corpus[[1]]))
#remove number
text_corpus <- tm_map(text_corpus, removeNumbers)
print(as.character(text_corpus[[1]]))
#remove whitespaces
text_corpus <- tm_map(text_corpus, stripWhitespace,lazy = T)
print(as.character(text_corpus[[1]]))
#remove stopwords
text_corpus <- tm_map(text_corpus, removeWords, c(stopwords('english')))
print(as.character(text_corpus[[1]]))

#convert to text document
text_corpus <- tm_map(text_corpus, PlainTextDocument)

#perform stemming - this should always be performed after text doc conversion
text_corpus <- tm_map(text_corpus, stemDocument,language = "english")
print(as.character(text_corpus[[1]]))
text_corpus[[1]]$meta #or
text_corpus[[1]]$content

#convert to document term matrix
docterm_corpus <- DocumentTermMatrix(text_corpus)

#find frequent terms
freq.out <- findFreqTerms(docterm_corpus, lowfreq = 100)
freq.out
colS <- colSums(as.matrix(docterm_corpus))
length(colS)

col_name <- attributes(colS)$names
col_name <- unlist(col_name)

col_sum <- data.table(name = attributes(colS)$names, number = colS)

#most frequent and least frequent words
colS[head(order(colS,decreasing = T))]

#remove sparse terms
new_docterm_corpus <- removeSparseTerms(docterm_corpus,sparse = 0.9999)

#visualize the data 
library(ggplot2)
wtf <- data.table(word = names(colS), freq = colS)

chart <- ggplot(subset(wtf, freq >10000), aes(x = word, y = freq))
chart <- chart + geom_bar(stat = 'identity', color = 'black', fill = 'lightblue')
chart <- chart + theme(axis.text.x=element_text(angle=45, hjust=1))
chart

findAssocs(new_docterm_corpus,terms = c("bathroom","build"),corlimit = 0.5)


#create wordcloud
library(wordcloud)
#this wordcloud is exported
wordcloud(names(colS), colS, min.freq = 100, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))
wordcloud(names(colS), colS, min.freq = 1000, scale = c(6,.1), colors = brewer.pal(6, 'Dark2'))

#craete data set for traning
data_mining <- as.data.table(as.matrix(new_docterm_corpus))

#TF IDF Data set
data_mining_tf <- as.data.table(as.matrix(weightTfIdf(new_docterm_corpus)))

#Word2Vec
#Word embedding
save.image("text_mining_3.RData")
load("text_mining.RData")


#after remove sparse terms, you can do clustering
#https://datascienceplus.com/extract-twitter-data-automatically-using-scheduler-r-package/

#create a new feature by clustering
#one sample matrix

terms <- new_docterm_corpus$dimnames$Terms



#create 2gram TDM
install.packages("RTextTools")
library(RTextTools)

ngram_corpus <- create_matrix(textColumns = train$description,ngramLength = 2,
                              removeNumbers = T
                              ,removePunctuation = T
                              ,removeStopwords = T
                              ,stemWords = T
                              ,stripWhitespace = T
                              ,toLower = T
                              )



f <- file("stdin")
open(f)

while(length(line <- readline(f, n=1)) > 0){
  write(line, stderr())
  dim(line)
}



#text to ngrams
install.packages("text2vec")
library(text2vec)
data("movie_review")
setDT(movie_review)

set.seed(2016L)
all_ids <- movie_review$id

train_ids <- sample(all_ids,4000)
test_ids <- setdiff(all_ids, train_ids)

setkey(movie_review,id)
setkey(movie_review,id)

train <- movie_review[J(train_ids)]
test <- movie_review[J(test_ids)]

#vocabulary based vectorization
prep_fun <- tolower
tok_fun <- word_tokenizer

it_train <- itoken(train$review
                   ,preprocessor = prep_fun
                   ,tokenizer = tok_fun
                   ,ids = train$id
                   ,progressbar = F)

#
vocab <- create_vocabulary(it_train)
vocab


#build a document term matrix
vectorizer <- vocab_vectorizer(vocab)

t1 <- Sys.time()

dtm_train <- create_dtm(it_train, vectorizer)
print(difftime(Sys.time(), t1, units = 'sec'))

dim(dtm_train)

#remove stop words
library(tidytext)

stop_vector <- stop_words$word

vocab <- create_vocabulary(it_train, stopwords = stop_vector)

#remove words occuring rarely
pruned_vocab <- prune_vocabulary(vocab, term_count_min = 5,doc_proportion_min = 0.001,doc_proportion_max = 0.5)
vectorizer <- vocab_vectorizer(pruned_vocab)

dtm_train <- create_dtm(it_train, vectorizer)
dim(dtm_train)

#use 2 grams
vocab <- create_vocabulary(it_train, ngram = c(1L,2L))

vocab <- prune_vocabulary(vocabulary = vocab, term_count_min = 10, doc_proportion_max = 0.5)

bigram_vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, bigram_vectorizer)
dim(dtm_train)


#Feature Hashing
h_vectorizer <- hash_vectorizer(hash_size = 2 ^ 14, ngram = c(1L,2L))

dtm_train <- create_dtm(it_train, h_vectorizer)


#normalization
dtm_train_l1_norm <- normalize(dtm_train, norm = "l1")

#TF IDF
vocab <- create_vocabulary(it_train)

vectorizer <- vocab_vectorizer(vocab)

dtm_train <- create_dtm(it_train, vectorizer)

#define tdidf nodel
tfidf <- TfIdf$new()

dtm_train_tfidf <- fit_transform(dtm_train, tfidf)

#https://cran.r-project.org/web/packages/text2vec/vignettes/text-vectorization.html

#machine learning with RtextTools

































