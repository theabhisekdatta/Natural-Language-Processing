# Load required libraries
library(readr)
library(dplyr)
library(tm)  # For text mining and TF-IDF
library(caret)  # For train-test split and model evaluation
library(glmnet)  # For Logistic Regression

# Load the data
data <- read_csv("IMDB Dataset.csv")

# Take the first 100 rows
df_100 <- head(data, 100)

# Text cleaning function
clean_text <- function(text) {
  text <- gsub("<.*?>", "", text)  # Remove HTML tags
  text <- gsub("[[:punct:]]", "", text)  # Remove punctuation
  text <- gsub("[0-9]+", "", text)  # Remove numbers
  text <- tolower(text)  # Convert to lowercase
  return(text)
}

# Apply the cleaning function to the review column
df_100$review <- sapply(df_100$review, clean_text)

# Split the data into features (X) and labels (y)
X <- df_100$review
y <- df_100$sentiment

# Train-test split
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex]
X_test <- X[-trainIndex]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]

cat("Training data shape:", length(X_train), length(y_train), "\n")
cat("Testing data shape:", length(X_test), length(y_test), "\n")

# Create a Corpus for both the training and testing sets
train_corpus <- Corpus(VectorSource(X_train))
test_corpus <- Corpus(VectorSource(X_test))

# Preprocessing control list
control <- list(weighting = weightTfIdf, stopwords = TRUE, removePunctuation = TRUE,
                removeNumbers = TRUE, tolower = TRUE)

# Create a DocumentTermMatrix for the training set
train_dtm <- DocumentTermMatrix(train_corpus, control = control)

# Use the terms from the training set to create a DTM for the test set
test_dtm <- DocumentTermMatrix(test_corpus, control = list(dictionary = Terms(train_dtm), 
                                                           weighting = weightTfIdf, 
                                                           stopwords = TRUE, 
                                                           removePunctuation = TRUE, 
                                                           removeNumbers = TRUE, 
                                                           tolower = TRUE))

# Convert DTMs to matrices
X_train_tfidf <- as.matrix(train_dtm)
X_test_tfidf <- as.matrix(test_dtm)

cat("TF-IDF transformed training data shape:", dim(X_train_tfidf), "\n")
cat("TF-IDF transformed testing data shape:", dim(X_test_tfidf), "\n")

# Logistic Regression using glmnet
model <- cv.glmnet(X_train_tfidf, as.factor(y_train), family = "binomial", alpha = 1, maxit = 100)

# Make predictions on the test set
y_pred <- predict(model, newx = X_test_tfidf, s = "lambda.min", type = "class")

# Calculate accuracy
accuracy <- sum(y_pred == y_test) / length(y_test)
cat("Accuracy:", accuracy, "\n")

# Generate a confusion matrix
conf_matrix <- caret::confusionMatrix(as.factor(y_pred), as.factor(y_test))
cat("Classification Report:\n")
print(conf_matrix)