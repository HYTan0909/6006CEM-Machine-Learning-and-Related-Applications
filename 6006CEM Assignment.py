# importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import nltk
import re
for dirname, _, filenames in os.walk('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning/assg/imdb-movie-reviews-dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from nltk.stem import WordNetLemmatizer

# reading the data from csv file
test_csv = pd.read_csv('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning'
                       '/assg/imdb-movie-reviews-dataset/test_data (1).csv')
train_csv = pd.read_csv('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning'
                        '/assg/imdb-movie-reviews-dataset/train_data (1).csv')

# Calculate review lengths for good reviews (label 1)
good_reviews = train_csv[train_csv['1'] == 1]
good_reviews['Review Length'] = good_reviews['0'].apply(lambda x: len(str(x).split()))

# Calculate review lengths for bad reviews (label 0)
bad_reviews = train_csv[train_csv['1'] == 0]
bad_reviews['Review Length'] = bad_reviews['0'].apply(lambda x: len(str(x).split()))

# Plotting histograms
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(good_reviews['Review Length'], bins=50, color='green', edgecolor='black')
plt.title('Word Count Distribution in Good Reviews')
plt.xlabel('Review Length')
plt.ylabel('Frequency')

plt.subplot(1, 2, 2)
plt.hist(bad_reviews['Review Length'], bins=50, color='red', edgecolor='black')
plt.title('Word Count Distribution in Bad Reviews')
plt.xlabel('Review Length')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

# stopword removal and lemmatization
stopwords = nltk.corpus.stopwords.words('english')
lemmatizer = WordNetLemmatizer()

# Preprocessing
nltk.download('stopwords')
train_csv.head()

train_X_non = train_csv['0']   # '0' refers to the review text
train_y = train_csv['1']   # '1' corresponds to Label (1 - positive and 0 - negative)
test_X_non = test_csv['0']
test_y = test_csv['1']

train_X=[]
test_X=[]

# text pre processing
for i in range(0, len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_X.append(review)

# text pre processing
for i in range(0, len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)

train_X[10]
test_X[10]

# tf idf
tf_idf = TfidfVectorizer()

# applying tf idf to training data
tf_train_x = tf_idf.fit_transform(train_X)

# applying tf idf to training data
tf_train_x = tf_idf.transform(train_X)
print("n_samples: %d, n_features: %d" % tf_train_x.shape)

# transforming test data into tf-idf matrix
tf_test_x = tf_idf.transform(test_X)
print("n_samples: %d, n_features: %d" % tf_test_x.shape)

# Naive Bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(tf_train_x, train_y)

# Evaluate Naive Bayes model
y_predict = naive_bayes_classifier.predict(tf_test_x)
print("Naive Bayes Classification Report:")
print(metrics.classification_report(test_y, y_predict, target_names=['Positive', 'Negative']))
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, y_predict))

# Define parameter grid for alpha (smoothing parameter)
param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5, 2.0]}

# Perform grid search
grid_search = GridSearchCV(naive_bayes_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(tf_train_x, train_y)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Use the best model for prediction
best_naive_bayes_classifier = grid_search.best_estimator_
best_naive_bayes_y_pred = best_naive_bayes_classifier.predict(tf_test_x)

# Evaluate the best Naive Bayes model
print("Best Naive Bayes Classification Report:")
print(metrics.classification_report(test_y, best_naive_bayes_y_pred, target_names=['Positive', 'Negative']))
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, best_naive_bayes_y_pred))

# doing a test prediction
test = ["this movie is amazing"]
review = re.sub('[^a-zA-Z]', ' ', test[0])
review = review.lower()
review = review.split()
review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
test_processed =[ ' '.join(review)]
print(test_processed)

test_input = tf_idf.transform(test_processed)
test_input.shape

# 0= bad review
# 1= good review
res = naive_bayes_classifier.predict(test_input)[0]
if res == 1:
    print("Good Review")
elif res == 0:
    print("Bad Review")

# Logistic Regression classifier
logistic_regression_classifier = LogisticRegression(solver='saga', max_iter=1000)
logistic_regression_classifier.fit(tf_train_x, train_y)

# Evaluate Logistic Regression model
logistic_regression_y_predict = logistic_regression_classifier.predict(tf_test_x)
print("Logistic Regression Classification Report:")
print(metrics.classification_report(test_y, logistic_regression_y_predict, target_names=['Positive', 'Negative']))
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, logistic_regression_y_predict))

# Define parameter grid for C (regularization parameter)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}

# Perform grid search
grid_search = GridSearchCV(logistic_regression_classifier, param_grid, cv=3, scoring='accuracy')
grid_search.fit(tf_train_x, train_y)

# Print the best parameters
print("Best parameters:", grid_search.best_params_)

# Use the best model for prediction
best_logistic_regression_classifier = grid_search.best_estimator_
best_logistic_regression_y_predict = best_logistic_regression_classifier.predict(tf_test_x)

# Evaluate the best Logistic Regression model
print("Best Logistic Regression Classification Report:")
print(metrics.classification_report(test_y, best_logistic_regression_y_predict, target_names=['Positive', 'Negative']))
print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, best_logistic_regression_y_predict))
