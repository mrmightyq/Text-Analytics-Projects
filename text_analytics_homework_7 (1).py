# -*- coding: utf-8 -*-
"""Text Analytics Homework 7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1myok1zdAI0hC-Y664MgoQrRpRWXS-tIj
"""

# Commented out IPython magic to ensure Python compatibility.
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from nltk.stem.snowball import SnowballStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
import pandas as pd 
import numpy as np
import re
import pandas as pd #data frame operations
import numpy as np #arrays and math functions
import matplotlib.pyplot as plt #2D plotting
# %matplotlib inline
import seaborn as sns #
import os
import io
from os import path
import re
from itertools import product
from scipy.stats import gaussian_kde as kde # for resampling dataset

# 
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

train=pd.read_csv("/content/train.tsv", delimiter='\t')
y=train['Sentiment'].values
X=train['Phrase'].values
train.head()

balance = train.groupby('Sentiment')['SentenceId'].count()
print(balance)

# Change: percent of total
round(balance.groupby(level=0).apply(lambda x:
                                                 100 * x / float(balance.sum())),2)

# Read the sklearn documentation to understand all vectorization options

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# several commonly used vectorizer setting

#  unigram boolean vectorizer, set minimum document frequency to 5
unigram_bool_vectorizer = CountVectorizer(encoding='latin-1', binary=True, min_df=5, stop_words='english')

#  unigram term frequency vectorizer, set minimum document frequency to 5
unigram_count_vectorizer = CountVectorizer(encoding='latin-1', binary=False, min_df=5, stop_words='english')

#  unigram and bigram term frequency vectorizer, set minimum document frequency to 5
gram12_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(1,2), min_df=5, stop_words='english')

#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer = TfidfVectorizer(encoding='latin-1', use_idf=True, min_df=5, stop_words='english')

#  bigram term frequency vectorizer, set minimum document frequency to 5
gram2_count_vectorizer = CountVectorizer(encoding='latin-1', ngram_range=(2,2), min_df=5, stop_words='english')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

"""CountVect"""

# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)
X_test_vec = unigram_count_vectorizer.transform(X_test)

"""Linear"""

# import the LinearSVC module
from sklearn.svm import LinearSVC
from sklearn.svm import SVC # for Support Vector Classification model

# initialize the LinearSVC model
svm_clf =  LinearSVC(C=1)

# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_vec,y_test)

from sklearn.metrics import confusion_matrix
y_pred = svm_clf.predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""RBF CountVec"""

svm_clf =  SVC(C=1,kernel='rbf')

# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_vec,y_test)

from sklearn.metrics import confusion_matrix
y_pred = svm_clf.predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""Poly"""

# initialize the LinearSVC model
svm_clf =  SVC(C=1,kernel='poly')

# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)

# test the classifier on the test data set, print accuracy score
svm_clf.score(X_test_vec,y_test)

# print confusion matrix and classification report

from sklearn.metrics import confusion_matrix
y_pred = svm_clf.predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""TF-idf"""

#  unigram tfidf vectorizer, set minimum document frequency to 5
unigram_tfidf_vectorizer
# fit vocabulary in training documents and transform the training documents into vectors
X_train_tf = unigram_tfidf_vectorizer.fit_transform(X_train)
X_test_tf = unigram_tfidf_vectorizer.transform(X_test)

# initialize the LinearSVC model
svm_clf =  LinearSVC(C=1)

# use the training data to train the model
svm_clf.fit(X_train_tf,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_tf,y_test)

from sklearn.metrics import confusion_matrix
y_pred = svm_clf.predict(X_test_tf)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

from sklearn.metrics import classification_report
target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""Poly tf-idf"""

# initialize the LinearSVC model
svm_clf =  SVC(C=1,kernel='poly')

# use the training data to train the model
svm_clf.fit(X_train_tf,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_tf,y_test)

y_pred = svm_clf.predict(X_test_tf)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""rbf tf-idf"""

# initialize the LinearSVC model
svm_clf =  SVC(C=1,kernel='rbf')

# use the training data to train the model
svm_clf.fit(X_train_tf,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_tf,y_test)

y_pred = svm_clf.predict(X_test_tf)
cm=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm)
print()

target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""Multinomial NB Count Vec"""

#Multinomial NB
multNB= MultinomialNB()

# use the training data to train the model
multNB.fit(X_train_vec,y_train)

# use the training data to train the model
multNB.fit(X_train_vec,y_train)

# print confusion matrix and classification report
multNB.fit(X_train_vec,y_train)
y_pred_mnb12 = multNB.predict(X_train_vec)
cm12=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm12)
print()

target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

"""Multinomial NB tf-idf"""

# use the training data to train the model
multNB.fit(X_train_tf,y_train)

# test the classifier on the test data set, print accuracy score

multNB.score(X_test_tf,y_test)

# print confusion matrix and classification report
multNB.fit(X_train_tf,y_train)
y_pred_mnb12 = multNB.predict(X_train_tf)
cm12=confusion_matrix(y_test, y_pred, labels=[0,1,2,3,4])
print(cm12)
print()

target_names = ['0','1','2','3','4']
print(classification_report(y_test, y_pred, target_names=target_names))

# Commented out IPython magic to ensure Python compatibility.
# Python program to generate WordCloud 

# %matplotlib inline
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in train.Phrase: 
	# typecaste each val to string 
	val = str(val) 

	# split the value 
	tokens = val.split() 
	
	# Converts each token into lowercase 
	for i in range(len(tokens)): 
		tokens[i] = tokens[i].lower() 
	
	comment_words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 800, height = 800, 
				background_color ='white', 
				stopwords = stopwords, 
				min_font_size = 10).generate(comment_words) 

# plot the WordCloud image					 
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad = 0) 

plt.show()

# initialize the LinearSVC model
svm_clf =  SVC(C=1.5,kernel='rbf')

# use the training data to train the model
svm_clf.fit(X_train_tf,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_tf,y_test)

svm_clf =  LinearSVC(C=5)

# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_vec,y_test)

svm_clf =  LinearSVC(C=100)

# use the training data to train the model
svm_clf.fit(X_train_vec,y_train)

# test the classifier on the test data set, print accuracy score

svm_clf.score(X_test_vec,y_test)