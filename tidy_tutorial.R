
path <- "/home/manish/Desktop/Data2017/February/twosigma/"
setwd(path)

textdata <- train$description[1:500]

library(dplyr)
library(tidytext)

do <- sample(x = 1:500, size = 500)
textdata <- train[do,.(description,interest_level)]

d <- textdata%>%unnest_tokens(word, description) #45193 rows

d1 <- d %>% anti_join(stop_words) #28716 rows

#check word frequency
d1%>%count(word, sort= T)

#create a plot for word frequency
d1%>%
  count(word, sort=TRUE)%>%
  filter(n > 200)%>%
  mutate(word = reorder(word, n))%>%
  ggplot(aes(word, n))+
  geom_bar(stat = 'identity',fill='lightblue',color='black')+
  xlab(NULL)+
  coord_flip()



#Novels of HGWells
library(gutenbergr)

hgwells <- gutenberg_download(c(35,36,5230,159))

tidy_hgwells <- hgwells%>%
  unnest_tokens(word,text)

tidy_hgwells <- tidy_hgwells%>%anti_join(stop_words)

tidy_hgwells%>%count(word, sort=T)


#SEntiment Analysis
sentiments
get_sentiments("nrc")

#use d1 here
nrcjoy <- get_sentiments("nrc")%>%
  filter(sentiment == "joy")

#join on words with joy sentiments
d1%>%
  inner_join(nrcjoy)

library(janeaustenr)
library(stringr)

tidy_books <- austen_books() %>%
  group_by(book) %>%
  mutate(linenumber = row_number(),
         chapter = cumsum(str_detect(text, regex("^chapter [\\divxlc]", 
                                                 ignore_case = TRUE)))) %>%
  ungroup() %>%
  unnest_tokens(word, text)

janeaustensentiment <- tidy_books %>%
  inner_join(get_sentiments("bing")) %>%
  count(book, index = linenumber %/% 100, sentiment) %>%
  spread(sentiment, n, fill = 0) %>%
  mutate(sentiment = positive - negative)


#on my train data

d1 <- d1%>%
  inner_join(get_sentiments("bing"))%>%
  mutate(linenumber = row_number())

#most common positive and ngeative word
d1 <- d1%>%
  inner_join(get_sentiments("bing"))%>%
  count(word, sentiment, sort=T)

#bar chart
d1%>%
  group_by(sentiment)%>%
  top_n(10)%>%
  mutate(word = reorder(word, n))%>%
  ggplot(aes(word, n, fill=sentiment))+
  geom_bar(stat = "identity")+
  facet_wrap(~sentiment)+
  labs(y = "Contribution to sentiment",x = NULL)+
  coord_flip()


d1 <- d1%>%
  count(interest_level, index = linenumber %/% 100, sentiment)%>%
  spread(sentiment,n,fill=0)%>%
  mutate(sentiment = positive - negative)

#positive, neative, net sentiment
ggplot(d1, aes(index, negative, fill=interest_level))+
  geom_bar(stat = "identity")+
  facet_wrap(~interest_level)


#word cloud
library(wordcloud)
d1%>%
  count(word)%>%
  with(wordcloud(word, n, max.words = 100))












