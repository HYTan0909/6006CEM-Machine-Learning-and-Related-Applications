#importing libraries
import numpy as np
import pandas as pd

import os
for dirname, _, filenames in os.walk('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning/assg/imdb-movie-reviews-dataset/'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

import nltk
nltk.download()
import re
import string
from nltk.stem import WordNetLemmatizer

#reading the data
test_csv = pd.read_csv('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning/assg/imdb-movie-reviews-dataset/test_data (1).csv')
train_csv = pd.read_csv('C:/Users/USER/Documents/Degree - BCSCUN/Y3S2 6006CEM Machine Learning/assg/imdb-movie-reviews-dataset/train_data (1).csv')

#stopword removal and lemmatization
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

#text pre processing
for i in range(0, len(train_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', train_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    train_X.append(review)

#text pre processing
for i in range(0, len(test_X_non)):
    review = re.sub('[^a-zA-Z]', ' ', test_X_non[i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords)]
    review = ' '.join(review)
    test_X.append(review)

train_X[10]
test_X[10]

#tf idf
tf_idf = TfidfVectorizer()

#applying tf idf to training data
X_train_tf = tf_idf.fit_transform(train_X)

#applying tf idf to training data
X_train_tf = tf_idf.transform(train_X)
print("n_samples: %d, n_features: %d" % X_train_tf.shape)

#transforming test data into tf-idf matrix
X_test_tf = tf_idf.transform(test_X)
print("n_samples: %d, n_features: %d" % X_test_tf.shape)

#naive bayes classifier
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier.fit(X_train_tf, train_y)

#predicted y
y_pred = naive_bayes_classifier.predict(X_test_tf)

print(metrics.classification_report(test_y, y_pred, target_names=['Positive', 'Negative']))

print("Confusion matrix:")
print(metrics.confusion_matrix(test_y, y_pred))

#doing a test prediction
test = ["this movie is totally terrible"]
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