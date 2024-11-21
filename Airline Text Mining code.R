# Install Important packages (Ensure these packages are already installed)
# install.packages("tm")
# install.packages("SnowballC")
# install.packages("dplyr")
# install.packages("nnet")
# install.packages("caTools")
# install.packages("Liblinear")
# install.packages("slam")
# install.packages("readr")
# install.packages("NLP")
# install.packages("lda")
# install.packages("LDAvis")
# devtools::install_github("cpsievert/LDAvisData")

# Load Important Libraries
library(tm)
library(SnowballC)
library(dplyr)
library(nnet)
library(caTools)
library(LiblineaR)
library(randomForest)
library(slam)
library(topicmodels)
library(wordcloud)
library(NLP)
library(LDAvis)

# Reading the data
airline_tweets <- read.csv('C:/Users/Dell/Desktop/7th Sem/NLP/review-2/twitterdata_airline.csv', stringsAsFactors = FALSE)

# Creating a subset with required variables for modeling
airline_tweets <- select(airline_tweets, airline_sentiment, Sentiment, airline, text)

# Create 2 subsets for positive and negative sentiment
positive_senti <- subset(airline_tweets, airline_sentiment == 'positive')
negative_senti <- subset(airline_tweets, airline_sentiment == 'negative')
dim(positive_senti)  # 2363 positive sentiments
dim(negative_senti)  # 9178 negative sentiments

###################################### TEXT PREPROCESSING START #########################################################
# Remove the unwanted symbols from text
airline_tweets$text <- gsub("^@\\w+ *", "", airline_tweets$text)

# Generate a function to analyze the corpus text
analyseText <- function(text) {
  negative_words <- c('bad', 'worse', 'terrible', 'horrible') # Placeholder, update as necessary
  words_in_text <- unlist(strsplit(tolower(text), "\\s+"))
  found_words <- words_in_text[words_in_text %in% negative_words]
  return(table(found_words))
}

library(tm)

# Create a corpus for collecting the text documents
corpus <- Corpus(VectorSource(airline_tweets$text))

# Define words to remove
wordsToRemove <- c('get', 'cant', 'can', 'now', 'just', 'will', 
                   'dont', 'ive', 'got', 'much', 'each', 'isnt', 
                   'unit', 'airline', 'virgin', 'southwestair', 
                   'americanair', 'usairway', 'jetblue', 'fli', 
                   'amp', 'dfw', 'tri', 'flt')

# Text Preprocessing
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, removeWords, wordsToRemove)
corpus <- tm_map(corpus, stemDocument)

# Inspect the cleaned corpus
inspect(corpus)

# Check if the corpus contains any documents before creating the DTM
if (length(corpus) > 0) {
  # Create a Document-Term Matrix
  dtm <- DocumentTermMatrix(corpus)
} else {
  cat("The corpus is empty after preprocessing.")
}


# Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)

# To inspect the document-term matrix 
inspect(dtm[1:5, 1:50])

# To find the most frequent terms, which are repeated more than 20 times
findFreqTerms(dtm, lowfreq = 20)

# Remove sparse terms from the document matrix
Sparse_senti <- removeSparseTerms(dtm, 0.98)

# Converting the sparse matrix to a data frame
SparseDF <- as.data.frame(as.matrix(Sparse_senti))
colnames(SparseDF) <- make.names(colnames(SparseDF))

SparseDF$sentiment <- airline_tweets$Sentiment

###################################### WORD CLOUDS START #########################################################
# To find the negative words after text preprocessing
negative_words <- analyseText(negative_senti$text)
print(negative_words)  # Check the frequency of each word

# To find the positive words after text preprocessing
positive_words <- analyseText(positive_senti$text)
print(positive_words)  # Check the frequency of each word

# Combine similar words (e.g., thank and thanks) if necessary

# Generate word clouds
par(mfrow = c(1, 2))

wordcloud(names(negative_words), freq = as.vector(negative_words), 
          random.order = FALSE, colors = brewer.pal(9, 'Reds')[4:9])

wordcloud(names(positive_words), freq = as.vector(positive_words), 
          random.order = FALSE, colors = brewer.pal(9, 'BuPu')[4:9])

###################################### LDA AND LDAVis CODE START #########################################################
# Prepare the data for LDA
# Load necessary libraries
# Load necessary libraries
library(tm)
library(topicmodels)

# Assuming your corpus is already created and cleaned
# Create a Document-Term Matrix
dtm <- DocumentTermMatrix(corpus)

# Check the dimensions of the DTM
print(dim(dtm))

# Inspect the DTM for non-zero entries
inspect(dtm)

# Find the indices of empty documents
empty_docs <- which(rowSums(as.matrix(dtm)) == 0)
print(empty_docs)  # List of empty document indices

# Filter out empty documents from the DTM
dtm <- dtm[rowSums(as.matrix(dtm)) > 0, ]

# LDA parameters
K <- 10  # Number of topics

# Run LDA using the topicmodels package
set.seed(357)  # For reproducibility
lda_model <- LDA(dtm, k = K)

# Inspect the results
terms(lda_model, 10)  # Get the top 10 terms for each topic
topics(lda_model)      # Get the topic assignments for each document




# Load necessary libraries
library(LDAvis)

# Get the topic-term distribution (phi) and document-topic distribution (theta)
phi <- posterior(lda_model)$terms  # Use posterior() to get the term-topic distribution
theta <- posterior(lda_model)$topics  # Use posterior() to get the document-topic distribution

# Normalize the phi matrix
phi <- t(apply(phi, 1, function(x) x / sum(x)))  # Normalize each row to sum to 1

# Convert to required format for LDAvis
json <- createJSON(phi = phi, 
                   theta = theta, 
                   doc.length = rowSums(as.matrix(dtm)), 
                   vocab = colnames(dtm), 
                   term.frequency = colSums(as.matrix(dtm)))

# Visualize using LDAvis
serVis(json, out.dir = 'visual', open.browser = interactive())



###################################### DATA MODELING START #########################################################
# Predicting the overall sentiments for airlines

# Modelling with multinomial regression
table(airline_tweets$Sentiment)

# Relevel sentiment to set baseline
SparseDF$sentiment <- as.factor(as.character(SparseDF$sentiment))
SparseDF$sentiment <- relevel(SparseDF$sentiment, ref = "-1")

# Split data into training and testing sets
set.seed(123)
split <- sample.split(airline_tweets$Sentiment, SplitRatio = 0.75)

Train_airline <- subset(SparseDF, split == TRUE)
Test_airline <- subset(SparseDF, split == FALSE)

# Multinomial Regression on Training Data
tweetmult <- multinom(sentiment ~ ., data = Train_airline)

# Predict on Training Data
predictmultinomTrain <- predict(tweetmult, newdata = Train_airline)

# Confusion Matrix for Training Data
table(Train_airline$sentiment, predictmultinomTrain)

# Accuracy on Training Data
train_accuracy <- sum(diag(table(Train_airline$sentiment, predictmultinomTrain))) / nrow(Train_airline)
print(train_accuracy)

# Predict on Testing Data
predictmultinomTest <- predict(tweetmult, newdata = Test_airline)

# Confusion Matrix for Testing Data
table(Test_airline$sentiment, predictmultinomTest)

# Accuracy on Testing Data
test_accuracy <- sum(diag(table(Test_airline$sentiment, predictmultinomTest))) / nrow(Test_airline)
print(test_accuracy)

# Using Liblinear Regularized Models to check accuracy
Train_Input <- Train_airline[, -ncol(Train_airline)]
Train_Decision <- Train_airline$sentiment

Test_Input <- Test_airline[, -ncol(Test_airline)]
Test_Decision <- Test_airline$sentiment

# Implement Liblinear for various models and check accuracy
Models <- 0:7
ModelCosts <- c(1000, 1, 0.001)
bestCost <- NA
bestACC <- 0
bestType <- NA

for (Model in Models) {
  for (costs in ModelCosts) {
    accuracy <- LiblineaR(Train_Input, target = Train_Decision, type = Model, cost = costs, 
                          bias = TRUE, cross = 5, verbose = FALSE)
    cat("Results for cost =", costs, ", accuracy =", accuracy, "\n")
    if (accuracy > bestACC) {
      bestACC <- accuracy
      bestCost <- costs
      bestType <- Model
    }
  }
}

# Train final model using best parameters
Final_Model <- LiblineaR(Train_Input, target = Train_Decision, type = bestType, cost = bestCost, bias = TRUE, verbose = FALSE)

# Prediction on Test Data
Prediction <- predict(Final_Model, Test_Input, proba = TRUE, decisionValues = TRUE)
head(Prediction$predictions)

# Check accuracy on Test Data
conf_matrix <- table(Test_Decision, Prediction$predictions)
Test_Accuracy <- sum(diag(conf_matrix)) / nrow(Test_airline)
print(Test_Accuracy)

