#!/usr/bin/env python
# coding: utf-8

# In[1]:


############################################################

# import packages for analysis and modeling
# import packages for analysis and modeling
import pandas as pd #data frame operations
import numpy as np #arrays and math functions
import matplotlib.pyplot as plt #2D plotting
get_ipython().run_line_magic('matplotlib', 'inline')
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
from sklearn.naive_bayes import MultinomialNB

#from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, precision_score, accuracy_score, make_scorer, classification_report, recall_score
from sklearn.model_selection import train_test_split, cross_val_score, cross_validate
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.ensemble import VotingClassifier

##################################


# In[2]:


## Step 1: Read in the file

## We cannot reac it in as csv because it is a mess

## One option is to convert it to text.



### !!!! NOTICE - I am using a VERY small sample of this data

### that I created by copying the column names and first 5

### rows into a new Excel - saving as .csv - and naming it as...

#RawfileName="SAMPLE_MovieReviewsFromSYRW2.csv"



### YOU MUST CHANGE THIS PATH or place your .csv file in the same

## location(folder) as your code. 

## First - we try this on SMALL EASY DATA

#RawfileName="C:/Users/profa/Documents/R/RStudioFolder_1/DrGExamples/R_For_DataScience/DATA/MovieDataSAMPLE_labeledVERYSMALL.csv"

## The larger version is here

## https://drive.google.com/file/d/1l3X_wd40Yokpbc-bEbiDkKfilMlP7AuQ/view?usp=sharing



## Then we use a larger dataset........

# datasets file names to load
deception_data = pd.read_csv(f'B:/GLOBAL/2063-BASUS/FLORHAM-PARK/NTH/NTH-T/TalentManagement/03_Data & Analytics (Quinn & Jake)/Quinn/Syracuse/Text Mining/Week 4/deception_data_converted_final.tsv', delimiter='\t')
df = deception_data
deception_data.head()


# In[3]:


# Python program to generate WordCloud 

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in df.review: 
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


# In[4]:


import numpy as np
from PIL import Image
import urllib
import requests
mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/2/b/c/5/1194984130298791075pizza_pepperoni.svg.med.png', stream=True).raw))

# This function takes in your text and your mask and generates a wordcloud. 
def generate_wordcloud(comment_words, mask):
    word_cloud = WordCloud(width = 512, height = 512, background_color='white', stopwords=STOPWORDS, mask=mask).generate(comment_words)
    plt.figure(figsize=(10,8),facecolor = 'white', edgecolor='blue')
    plt.imshow(word_cloud)
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()
    
#Run the following to generate your wordcloud
generate_wordcloud(comment_words, mask)


# In[5]:


import random
def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 250)

wc = WordCloud(max_words=1000, mask=mask, stopwords=stopwords, margin=10,
               random_state=1).generate(comment_words)
# store default colored image
default_colors = wc.to_array()
plt.title("Custom colors")
plt.imshow(wc.recolor(color_func=grey_color_func, random_state=3),
           interpolation="bilinear")
wc.to_file("a_new_hope.png")
plt.axis("off")
plt.figure()
plt.title("Default colors")
plt.imshow(default_colors, interpolation="bilinear")
plt.axis("off")
plt.show()


# In[6]:


df_sent = df.drop(columns=["lie"])
df_sent.head()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[7]:


## Let's build a small function that will find 

## numbers/digits and return True if so



##------------------------------------------------------

### DEFINE A FUNCTION that returns True if numbers

##  are in a string 

def Logical_Numbers_Present(anyString):

    return any(char.isdigit() for char in anyString)


# In[8]:



    ## NOTE: You can also use this code to CONTROL


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[9]:


###########################################################

##

##    Naive Bayes 

##

###########################################################

from sklearn.naive_bayes import MultinomialNB as NB



# In[ ]:





# In[ ]:





# In[ ]:





# In[10]:


## Create the confusion matrix

from sklearn.metrics import confusion_matrix


# In[11]:


## prettier-------plot...............

import sklearn.metrics

## If the following line does not work - 

## run this:

## conda update -c conda-forge scikit-learn



from sklearn.metrics import confusion_matrix

import numpy as np

import matplotlib.pyplot as plt

import itertools









## REF:

##https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix with Naive Bayes',

                          cmap=plt.cm.Reds):

    

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=26)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45,fontsize=26)

    plt.yticks(tick_marks, classes,fontsize=26)



    



    print(cm)

    print(cm.shape[0])



 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(i, j, cm[i,j],fontsize=26,

                 horizontalalignment="center",

                 color="gray")



    plt.tight_layout()

    plt.ylabel('True label',fontsize=26)

    plt.xlabel('Predicted label',fontsize=26)

    #----------------------------------------end of function


# In[ ]:





# In[12]:


# evaluate a NB model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score


# In[ ]:





# In[ ]:





# In[13]:


import numpy as np
import pandas as pd


# In[14]:



import matplotlib.pyplot as pls 


# In[15]:


################ Sentiment Now ####################################
df_sent.to_csv(r'C:/Users/KnudseQ/Desktop/Sent_File.csv', index = False)
len(df_sent)


# In[16]:


RawfileName = 'C:/Users/KnudseQ/Desktop/Sent_File.csv'
file = open(RawfileName, 'r') # 'r' is for 'read' permissions
file_data = [row for row in file]
print(file_data)


# In[17]:


AllReviewsList=[]   #content

AllLabelsList=[]    #labels
 


# In[18]:


with open(RawfileName,'r') as FILE:    # "a", "w"

    FILE.readline() # skip header line - skip row 1

    ## This reads the line and so does nothing with it

    for row in FILE:   #starts on row 2

        print(row)

        #print(type(row))

        NextLabel,NextReview=row.split(",", 1)

        #print("The label is:\n", NextLabel, "\n")

        #print(NextReview)

        AllReviewsList.append(NextReview)

        AllLabelsList.append(NextLabel)

 ##----------------------------------------   


# In[19]:


print(AllReviewsList)  #List of content

print(AllLabelsList)


# In[20]:


df = pd.DataFrame(AllReviewsList)
df.rename(columns={ df.columns[0]: "Review" }, inplace = True)
display(df)


# In[21]:


My_CV1=CountVectorizer(input='content',

                        stop_words='english',

                        #max_features=100                       

                        )


# In[22]:


custom_stopwords = ["abc"]
stopwords = set(STOPWORDS).union(custom_stopwords) 
My_TF1=TfidfVectorizer(input='content',

                        stop_words='english',

                        #max_features=100                      

                        )

## NOw I can vectorize using my list of complete paths to my files

X_CV1=My_CV1.fit_transform(AllReviewsList)

X_TF1=My_TF1.fit_transform(AllReviewsList)



print(My_CV1.vocabulary_)

print(My_TF1.vocabulary_)


# In[23]:


ColNames=My_TF1.get_feature_names()



## OK good - but we want a document topic model A DTM (matrix of counts)

DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)

DataFrame_TF=pd.DataFrame(X_TF1.toarray(), columns=ColNames)



## Drop/remove columns not wanted

print(DataFrame_CV.columns)

print(DataFrame_CV)

print(DataFrame_TF)


# In[24]:


for nextcol in DataFrame_CV.columns:

    #print(nextcol)

    ## Remove unwanted columns

    #Result=str.isdigit(nextcol) ## Fast way to check numbers

    #print(Result)

    

    ##-------------call the function -------

    LogResult=Logical_Numbers_Present(nextcol)

    #print(LogResult)

    ## The above returns a logical of True or False

    

    ## The following will remove all columns that contains numbers

    if(LogResult==True):

        #print(LogResult)

        #print(nextcol)

        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)

        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)



    ## The following will remove any column with name

    ## of 3 or smaller - like "it" or "of" or "pre".

    ##print(len(nextcol))  ## check it first

    ## NOTE: You can also use this code to CONTROL

    ## the words in the columns. For example - you can

    ## have only words between lengths 5 and 9. 

    ## In this case, we remove columns with words <= 3.

    elif(len(str(nextcol))<=2):

        print(nextcol)

        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)

        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)

    

    

print(DataFrame_CV)

print(DataFrame_TF)


# In[25]:


## Recall:

print(AllLabelsList)

print(type(AllLabelsList))


# In[26]:


## Place these on dataframes:

## List --> DF

DataFrame_CV.insert(loc=0, column='LABEL', value=AllLabelsList)

DataFrame_TF.insert(loc=0, column='LABEL', value=AllLabelsList)

print(DataFrame_CV)

print(DataFrame_TF)


# In[27]:


DataFrame_TF.to_csv(r'C:/Users/KnudseQ/Desktop/Sent_File_Clean.csv', index = False)
#len(DataFrame_TF)
#len(AllLabelsList)


# In[28]:


###########################################################

##

##    Naive Bayes 

##

###########################################################

from sklearn.naive_bayes import MultinomialNB as NB



MyNB=NB(alpha=1)



## The data is not ready to analyze yet

## We first need to break it onto training and testing 



## There are many ways to this - sampling - sampling

## with or without replacement - random and non-random

## etc. I will use sklears train/test package



from sklearn.model_selection import train_test_split as TTS



NB_train, NB_test=TTS(DataFrame_CV, test_size=0.33, random_state=42)
print(NB_train, "/n")

print(NB_test, "/n")


# In[29]:


## Now we need to REMOVE THE LABELS and save them!

## !!!!!!!!! Do not forget this step !!!!!!!!!!!!!



NB_TrainLABEL=NB_train["LABEL"]

## Remove the label from the data

NB_TrainDATA=NB_train.drop(columns="LABEL")

print(NB_TrainLABEL)

print(NB_TrainDATA)


# In[30]:


NB_TestLABEL=NB_test["LABEL"]

NB_TestDATA=NB_test.drop(columns="LABEL")

print(NB_TestLABEL)

print(NB_TestDATA)


# In[31]:


##OK - now we can fit our NB model.

#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

MyNB.fit(NB_TrainDATA, NB_TrainLABEL)



## NOw - see how your model did:

NB_Model_predictions=MyNB.predict(NB_TestDATA)

print("The model predicted the following\n")

print(NB_Model_predictions)



print("The actual labels are\n")
print(NB_TestLABEL)


# In[32]:


## Create the confusion matrix

from sklearn.metrics import confusion_matrix

confusion_matrix(NB_TestLABEL, NB_Model_predictions)


# In[33]:


## prettier-------plot...............

import sklearn.metrics

## If the following line does not work - 

## run this:

## conda update -c conda-forge scikit-learn



from sklearn.metrics import confusion_matrix

import numpy as np

import matplotlib.pyplot as plt

import itertools







MY_cnf_matrix = confusion_matrix(NB_TestLABEL, NB_Model_predictions)



## REF:

##https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix with Naive Bayes',

                          cmap=plt.cm.Reds):

    

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=26)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45,fontsize=26)

    plt.yticks(tick_marks, classes,fontsize=26)



    



    print(cm)

    print(cm.shape[0])



 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(i, j, cm[i,j],fontsize=26,

                 horizontalalignment="center",

                 color="gray")



    plt.tight_layout()

    plt.ylabel('True label',fontsize=26)

    plt.xlabel('Predicted label',fontsize=26)

    #----------------------------------------end of function


# In[34]:


# Compute confusion matrix

cnf_matrix = confusion_matrix(NB_TestLABEL, NB_Model_predictions)

np.set_printoptions(precision=2)



class_names=["Positive", "Negative"]



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix for Naive Bayes')





plt.show()



# In[35]:


# evaluate a NB model using k-fold cross-validation
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
# prepare the cross-validation procedure
X, y = NB_TrainDATA, NB_TrainLABEL 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = NB(alpha=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[36]:


scores = cross_val_score(NB(alpha=1), X, y, cv=10)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[37]:


model.fit(X, y)
#imps = permutation_importance(model, X, y)
#print(imps.importances_mean)
features = X.columns
feat=pd.DataFrame(features)
print(feat)


# In[38]:


print(model.coef_)
print(features)


# In[39]:


features = X.columns
feat=pd.DataFrame(features)


# In[40]:


import numpy as np
import pandas as pd
coff=pd.DataFrame(model.coef_)
coff=coff.transpose()


# In[41]:


model_imp=pd.concat([coff, feat], axis=1)
model_imp.columns =['Importance', 'Feature']
#sort dataframe
sorted_df = model_imp.sort_values(by='Importance', ascending=False)
sorted_df[0:20]


# In[42]:


top20feat = sorted_df[0:20]


# 

# In[43]:


import matplotlib.pyplot as pls 
top20feat.plot(x='Feature', y='Importance', kind='bar') 
plt.show()


# In[44]:


# create model
from sklearn.naive_bayes import BernoulliNB
model = BernoulliNB()
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[45]:


NB_train, NB_test=TTS(DataFrame_TF, test_size=0.33, random_state=42)
print(NB_train, "/n")

print(NB_test, "/n")


# In[46]:


## Now we need to REMOVE THE LABELS and save them!

## !!!!!!!!! Do not forget this step !!!!!!!!!!!!!



NB_TrainLABEL=NB_train["LABEL"]

## Remove the label from the data

NB_TrainDATA=NB_train.drop(columns="LABEL")

print(NB_TrainLABEL)

print(NB_TrainDATA)


# In[47]:


NB_TestLABEL=NB_test["LABEL"]

NB_TestDATA=NB_test.drop(columns="LABEL")

print(NB_TestLABEL)

print(NB_TestDATA)


# In[48]:


##OK - now we can fit our NB model.

#https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html#sklearn.naive_bayes.MultinomialNB

MyNB.fit(NB_TrainDATA, NB_TrainLABEL)



## NOw - see how your model did:

NB_Model_predictions=MyNB.predict(NB_TestDATA)

print("The model predicted the following\n")

print(NB_Model_predictions)



print("The actual labels are\n")
print(NB_TestLABEL)


# In[49]:


confusion_matrix(NB_TestLABEL, NB_Model_predictions)


# In[50]:


# Compute confusion matrix

cnf_matrix = confusion_matrix(NB_TestLABEL, NB_Model_predictions)

np.set_printoptions(precision=2)



class_names=["Positive", "Negative"]



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix for Naive Bayes')





plt.show()



# In[51]:


# prepare the cross-validation procedure
X, y = NB_TrainDATA, NB_TrainLABEL 
cv = KFold(n_splits=10, random_state=1, shuffle=True)
# create model
model = NB(alpha=1)
# evaluate model
scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
# report performance
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))


# In[52]:


scores = cross_val_score(NB(alpha=1), X, y, cv=10)
scores
print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))


# In[53]:


model.fit(X, y)
#imps = permutation_importance(model, X, y)
#print(imps.importances_mean)
features = X.columns
feat=pd.DataFrame(features)
print(feat)


# In[54]:


coff=pd.DataFrame(model.coef_)
coff=coff.transpose()


# In[55]:


model_imp=pd.concat([coff, feat], axis=1)
model_imp.columns =['Importance', 'Feature']
#sort dataframe
sorted_df = model_imp.sort_values(by='Importance', ascending=False)
sorted_df[0:20]


# In[56]:


top20feat = sorted_df[0:20]
top20feat.plot(x='Feature', y='Importance', kind='bar') 
plt.show()


# In[57]:


from sklearn.naive_bayes import BernoulliNB

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


# In[58]:


train =df_sent
y=train['sentiment'].values
X=train['review'].values


# In[88]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
print(X_train[0])
print(y_train[0])
print(X_test[0])
print(y_test[0])


# In[89]:


unique, counts = np.unique(y_train, return_counts=True)
print(np.asarray((unique, counts)))


# In[90]:


# Print out the category distribution in the test data set. 
#Is the test data set's category distribution similar to the training data set's?

# Your code starts here
unique, counts = np.unique(y_test, return_counts=True)
print(np.asarray((unique, counts)))
# Your code ends here


# In[91]:


# The vectorizer can do "fit" and "transform"
# fit is a process to collect unique tokens into the vocabulary
# transform is a process to convert each document to vector based on the vocabulary
# These two processes can be done together using fit_transform(), or used individually: fit() or transform()

# fit vocabulary in training documents and transform the training documents into vectors
X_train_vec = unigram_count_vectorizer.fit_transform(X_train)


# In[92]:


# use the vocabulary constructed from the training data to vectorize the test data. 
# Therefore, use "transform" only, not "fit_transform", 
# otherwise "fit" would generate a new vocabulary from the test data

X_test_vec = unigram_count_vectorizer.transform(X_test)


# In[93]:


# initialize the MNB model
nb_clf= MultinomialNB()

# use the training data to train the MNB model
nb_clf.fit(X_train_vec,y_train)


# In[94]:


# sort the conditional probability for category 0 "very negative"
# print the words with highest conditional probs
# these can be words popular in the "very negative" category alone, or words popular in all cateogires

feature_ranks = sorted(zip(nb_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
very_negative_features = feature_ranks[-10:]
print(very_negative_features)


# In[95]:


# test the classifier on the test data set, print accuracy score

nb_clf.score(X_test_vec,y_test)


# In[96]:


# print confusion matrix (row: ground truth; col: prediction)

from sklearn.metrics import confusion_matrix
y_pred = nb_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=['n','p'])
print(cm)


# In[97]:


# print classification report

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))

from sklearn.metrics import classification_report
target_names = ['neg','pos']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[98]:


## find the calculated posterior probability
posterior_probs = nb_clf.predict_proba(X_test_vec)

## find the posterior probabilities for the first test example
print(posterior_probs[0])

# find the category prediction for the first test example
y_pred = nb_clf.predict(X_test_vec)


# In[99]:


X_train_vec_bool = unigram_bool_vectorizer.fit_transform(X_train)
bernoulliNB_clf = BernoulliNB(X_train_vec_bool, y_train)


# In[109]:


# cross validation

from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
b_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=True)),('nb', BernoulliNB())])
scores = cross_val_score(b_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("BernouilliNB: Count Vect, binary=True:",avg)

b_clf_pipe = Pipeline([('vect',  TfidfVectorizer(encoding='latin-1', binary=True)),('nb', BernoulliNB())])
scores = cross_val_score(b_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("BernouilliNB: Tf-idf, binary=True:",avg)


# In[101]:


b_clf_pipe.fit(X, y)
#imps = permutation_importance(model, X, y)
#print(imps.importances_mean)
feature_ranks = sorted(zip(b_clf.feature_log_prob_[0], unigram_count_vectorizer.get_feature_names()))
very_negative_features = feature_ranks[-10:]
print(very_negative_features)


# In[118]:


# test the classifier on the test data set, print accuracy score
# initialize the MNB model
b_clf= BernoulliNB()
# use the training data to train the MNB model
b_clf.fit(X_train_vec,y_train)
b_clf.score(X_test_vec,y_test)
y_pred = b_clf.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=['n','p'])
print(cm)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))

from sklearn.metrics import classification_report
target_names = ['neg','pos']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[119]:


## prettier-------plot...............

import sklearn.metrics

## If the following line does not work - 

## run this:

## conda update -c conda-forge scikit-learn



from sklearn.metrics import confusion_matrix

import numpy as np

import matplotlib.pyplot as plt

import itertools







MY_cnf_matrix = confusion_matrix(NB_TestLABEL, NB_Model_predictions)



## REF:

##https://scikit-learn.org/0.18/auto_examples/model_selection/plot_confusion_matrix.html



def plot_confusion_matrix(cm, classes,

                          normalize=False,

                          title='Confusion matrix with Naive Bayes',

                          cmap=plt.cm.Reds):

    

    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    plt.title(title, fontsize=26)

    plt.colorbar()

    tick_marks = np.arange(len(classes))

    plt.xticks(tick_marks, classes, rotation=45,fontsize=26)

    plt.yticks(tick_marks, classes,fontsize=26)



    



    print(cm)

    print(cm.shape[0])



 

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        plt.text(i, j, cm[i,j],fontsize=26,

                 horizontalalignment="center",

                 color="gray")



    plt.tight_layout()

    plt.ylabel('True label',fontsize=26)

    plt.xlabel('Predicted label',fontsize=26)

    #----------------------------------------end of function


# In[121]:


# Compute confusion matrix

cnf_matrix = cm=confusion_matrix(y_test, y_pred, labels=['n','p'])

np.set_printoptions(precision=2)



class_names=["Positive", "Negative"]



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix for BernoulliNB')





plt.show()


# In[105]:


# run 3-fold cross validation to compare the performance of 
# (1) BernoulliNB (2) MultinomialNB with TF vectors (3) MultinomialNB with boolean vectors

# Your code starts here
nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("MNB: count vect, binary=False:",avg)

nb_clf_pipe = Pipeline([('vect', CountVectorizer(encoding='latin-1', binary=True)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("MNB: Count vect, binary=True:",avg)

nb_clf_pipe = Pipeline([('vect', unigram_tfidf_vectorizer),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("MNB: Tf-idf, binary=True:",avg)


nb_clf_pipe = Pipeline([('vect', gram12_count_vectorizer),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("MNB: gram12 count vect, binary=False:",avg)

nb_clf_pipe = Pipeline([('vect', TfidfVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])
scores = cross_val_score(nb_clf_pipe, X, y, cv=10)
avg=sum(scores)/len(scores)
print("MNB: Tf-idf, binary=False:",avg)

# Your code ends here


# In[116]:


# test the classifier on the test data set, print accuracy score
# initialize the MNB model
b_clf_pipe= Pipeline([('vect', TfidfVectorizer(encoding='latin-1', binary=False)),('nb', MultinomialNB())])
# use the training data to train the MNB model
b_clf_pipe.fit(X_train_vec,y_train)
b_clf_pipe.score(X_test_vec,y_test)
y_pred = b_clf_pipe.fit(X_train_vec, y_train).predict(X_test_vec)
cm=confusion_matrix(y_test, y_pred, labels=['n','p'])
print(cm)
print(precision_score(y_test, y_pred, average=None))
print(recall_score(y_test, y_pred, average=None))

from sklearn.metrics import classification_report
target_names = ['neg','pos']
print(classification_report(y_test, y_pred, target_names=target_names))


# In[115]:


MY_cnf_matrix = confusion_matrix(NB_TestLABEL, NB_Model_predictions)
cnf_matrix = cm=confusion_matrix(y_test, y_pred, labels=['n','p'])

np.set_printoptions(precision=2)



class_names=["Positive", "Negative"]



# Plot non-normalized confusion matrix

plt.figure()

plot_confusion_matrix(cnf_matrix, classes=class_names,

                      title='Confusion matrix for Naive Bayes')





plt.show()


# In[106]:


from sklearn.feature_extraction.text import CountVectorizer
import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')
class StemmedCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(StemmedCountVectorizer, self).build_analyzer()
        return lambda doc: ([english_stemmer.stem(w) for w in analyzer(doc)])

stem_vectorizer = StemmedCountVectorizer(min_df=3, analyzer="word")
X_train_stem_vec = stem_vectorizer.fit_transform(X_train)
#y_train_stem_vec = stem_vectorizer.fit_transform(y_train)


# In[107]:


# check the content of a document vector
print(X_train_stem_vec.shape)
print(X_train_stem_vec[0].toarray())

# check the size of the constructed vocabulary
print(len(stem_vectorizer.vocabulary_))

# print out the first 10 items in the vocabulary
print(list(stem_vectorizer.vocabulary_.items())[:10])


# In[108]:


# initialize the MNB model
nb_clf= MultinomialNB()

# use the training data to train the MNB model
nb_clf.fit(X_train_stem_vec,y_train)


# In[ ]:





# In[ ]:





# In[ ]:




