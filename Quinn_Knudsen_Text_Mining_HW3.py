#!/usr/bin/env python
# coding: utf-8

# In[143]:


############################################################

##

## WORKING WITH TEXT IN PYTHON - CSV Files with clear labels

##

## Reading in and vectorizing

## various formats for text data

##

## This example shows what to do with 

## a very poorly formatted and dirty 

## csv file.

## 

## RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE

## HERE

# https://drive.google.com/file/d/11H6AbWxKsPLY3yt__OrmK0rjjYShKhig/view?usp=sharing

#########################################

## Great tutorial

## https://kavita-ganesan.com/how-to-use-countvectorizer/#.XpIWwXJ7nb0

## Textmining Naive Bayes Example

import nltk

import pandas as pd

import sklearn

import re  

from sklearn.feature_extraction.text import CountVectorizer



#Convert a collection of raw documents to a matrix of TF-IDF features.

#Equivalent to CountVectorizer but with tf-idf norm

from sklearn.feature_extraction.text import TfidfVectorizer





from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

## For Stemming

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import os



from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

import string



##################################



## In this case, our lives are easier

## because the label in this csv is 

## in the first column and is easy to find

## and to save.

##

## So, let's try an easier method for this

## cleaner csv file.

## Note however that this data still has the review

## portions in multiple columns. 

## When we read it in, these will translate into commas

## This is good - it will allow us to split by comma

## save the label, and prepare the data. 



########################################################



### !!!! NOTICE - I am using a VERY small sample of this data



### YOU MUST CHANGE THIS PATH or place your .csv file in the same

## location(folder) as your code. 


# In[63]:


# filename = 'moviereviewRAW.csv'
RawfileName = 'C:/Users/KnudseQ/Desktop/Homework 3/RestaurantSentimentCleanerLABELEDDataSMALLSAMPLE.csv'
file = open(RawfileName, 'r') # 'r' is for 'read' permissions
file_data = [row for row in file]
print(file_data)


# In[ ]:





# In[64]:


## This file has a header. 

## It has "setinment" and "review" on the first row.

## This will be treated like all other text and that is BAD - why?

## We need to remove it first.

## There are many ways to do this. 

## We can use seek(), we can skip it with a counter, etc. 

## But, the best way is by using "with open" and readline()



## Because the label is in the first column, we can split the

## string into two parts  so 1 split  after the first comma....

## We will create a list of labels and a list of reviews

AllReviewsList=[]   #content

AllLabelsList=[]    #labels


# In[65]:


#-----------------for loop---------------



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


# In[66]:


print(AllReviewsList)  #List of content

print(AllLabelsList)


# In[105]:


from wordcloud import WordCloud, STOPWORDS 
import nltk
from nltk.corpus import stopwords
df = pd.DataFrame(AllReviewsList)
df.rename(columns={ df.columns[0]: "Review" }, inplace = True)
display(df)


# In[128]:


# Python program to generate WordCloud 

get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in df.Review: 
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


# In[129]:


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


# In[88]:


########################################

##

## CountVectorizer  and TfidfVectorizer

##

########################################

## Now we have what we need!

## We have a list of the contents (reviews)

## in the csv file.



My_CV1=CountVectorizer(input='content',

                        stop_words='english',

                        #max_features=100                       

                        )


# In[68]:


My_TF1=TfidfVectorizer(input='content',

                        stop_words='english',

                        #max_features=100                      

                        )

## NOw I can vectorize using my list of complete paths to my files

X_CV1=My_CV1.fit_transform(AllReviewsList)

X_TF1=My_TF1.fit_transform(AllReviewsList)



print(My_CV1.vocabulary_)

print(My_TF1.vocabulary_)


# In[69]:


## Hmm - that's not quite what we want...

## Let's get the feature names which ARE the words

## The column names are the same for TF and CV

ColNames=My_TF1.get_feature_names()



## OK good - but we want a document topic model A DTM (matrix of counts)

DataFrame_CV=pd.DataFrame(X_CV1.toarray(), columns=ColNames)

DataFrame_TF=pd.DataFrame(X_TF1.toarray(), columns=ColNames)



## Drop/remove columns not wanted

print(DataFrame_CV.columns)

print(DataFrame_CV)

print(DataFrame_TF)


# In[70]:


## Let's build a small function that will find 

## numbers/digits and return True if so



##------------------------------------------------------

### DEFINE A FUNCTION that returns True if numbers

##  are in a string 

def Logical_Numbers_Present(anyString):

    return any(char.isdigit() for char in anyString)


# In[71]:


##----------------------------------------------------



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

    elif(len(str(nextcol))<=3):

        print(nextcol)

        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)

        DataFrame_TF=DataFrame_TF.drop([nextcol], axis=1)

    

    

print(DataFrame_CV)

print(DataFrame_TF)


# In[72]:


##################################################

##

## Alternative code for the above...........

##

###################################################

#for nextcol in DataFrame_CV.columns:

#    if(re.search(r'[^A-Za-z]+', nextcol)):

#        print(nextcol)

#        DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)

#    elif(len(str(nextcol))<=3):

#        print(nextcol)

#       DataFrame_CV=DataFrame_CV.drop([nextcol], axis=1)


# In[73]:


####################################################

##

## Adding the labels to the data......

## This would be good if you are modeling with

## supervised methods - such as NB, SVM, DT, etc.

##################################################

## Recall:

print(AllLabelsList)

print(type(AllLabelsList))



# In[74]:


## Place these on dataframes:

## List --> DF

DataFrame_CV.insert(loc=0, column='LABEL', value=AllLabelsList)

DataFrame_TF.insert(loc=0, column='LABEL', value=AllLabelsList)

print(DataFrame_CV)

print(DataFrame_TF)


# In[75]:



############################################

##

##  WRITE CLEAN, Tokenized, vectorized data

##  to new csv file. This way,  you can read it

##  into any program and work with it.

##

######################################################







CV_File="MyTextOutfile_count.csv"

TF_File="MyTextOutfile_Tfidf.csv"




# In[76]:


######## Method 1: Advanced ---------------



## This is commented out - but you can uncomment to play with it

#import tkinter as tk

#from tkinter import filedialog

# =============================================================================

# root= tk.Tk()

# 

# def exportCSV(df):

#     #global df

#     export_file_path = filedialog.asksaveasfilename(defaultextension='.csv')

#     df.to_csv(export_file_path, index = False, header=True)

# 

# 

# USER_Button_CSV = tk.Button(text='File Saved - Close this Box', 

#                              ## Call the function here...

#                              command=exportCSV(DataFrame_CV), 

#                              bg='blue', 

#                              fg='grey', 

#                              font=('helvetica', 11, 'bold'))

# 

# USER_Button_CSV.pack(side=tk.LEFT)

# 

# root.mainloop()

# =============================================================================


# In[85]:


################ Method 2: Save csv directly --



DataFrame_CV.to_csv(r'C:/Users/KnudseQ/Desktop/Homework 3/CV_File.csv', index = False)

DataFrame_TF.to_csv(r'C:/Users/KnudseQ/Desktop/Homework 3/TF_File.csv', index = False)


# In[81]:


display(DataFrame_TF)


# In[ ]:





# In[144]:


############################################################

##

## WORKING WITH TEXT IN PYTHON - CSV Files

##

## Reading in and vectorizing

## various formats for text data

##

## This example shows what to do with 

## a very poorly formatted and dirty 

## csv file.

## I will use the MovieReviews csv file. 

## Here is a link to the raw and original

## file: MovieReviewsFromSYRW2.csv which is HERE...

## USE A SMALL SAMPLE OF THIS - this file is BIG :)

## https://drive.google.com/file/d/1KgycYN1G4zU9IHscZWDTiAn7j-qg-aIz/view?usp=sharing

#########################################



## Textmining Naive Bayes Example

import nltk

import pandas as pd

import sklearn

import re  

from sklearn.feature_extraction.text import CountVectorizer



#Convert a collection of raw documents to a matrix of TF-IDF features.

#Equivalent to CountVectorizer but with tf-idf norm

from sklearn.feature_extraction.text import TfidfVectorizer





from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

import matplotlib.pyplot as plt

from nltk.corpus import stopwords

## For Stemming

from nltk.stem import PorterStemmer

from nltk.tokenize import sent_tokenize, word_tokenize

import os



from nltk.stem.wordnet import WordNetLemmatizer

from nltk.stem.porter import PorterStemmer

import string



# In[149]:


##################################



## Step 1: Read in the file

## We cannot reac it in as csv because it is a mess

## One option is to convert it to text.



### !!!! NOTICE - I am using a VERY small sample of this data

### that I created by copying the column names and first 5

### rows into a new Excel - saving as .csv - and naming it as...

#RawfileName="SAMPLE_MovieReviewsFromSYRW2.csv"



### YOU MUST CHANGE THIS PATH or place your .csv file in the same

## location(folder) as your code. 

RawfileName="C:/Users/KnudseQ/Desktop/Homework 3/MovieDataSAMPLE_labeledVERYSMALL.csv"





FILE=open(RawfileName,"r")



## We are going to clean it and then write it back to csv!

## So, we need an empty csv file - let's make one....

filename="CleanText4.csv"

NEWFILE=open(filename,"w")

## In the first row, create a column called Label and a column Text...

ToWrite="Label,Text\n"

## Write this to new empty cs v file

NEWFILE.write(ToWrite)

## Close it up

NEWFILE.close()


# In[150]:


### Now, we have an empty csv file called CLeanText.csv

### Above we created the first row of column names: Label and Text

### Next, we will open this file for "a" or append - so we can

### add things to it from where we left off

### NOTE: If you open this file again with "w" it will write over

### whatever is in the file!  USE "a"....

### This line of code opens the file for append and creates

### a variable (NEWFILE) that we can use to access and control the

### file.

NEWFILE=open(filename, "a")



### We also will build a CLEAN dataframe.

### So for now, we need a blank one...

MyFinalDF=pd.DataFrame()



# In[151]:


###########

## Read the new csv file you created into a DF or into CounterVectorizer

#######

## recall that filename is CleanFile.csv - the file we just made

## Into DF

MyTextDF=pd.read_csv(filename)

## remove any rows with NA

MyTextDF = MyTextDF.dropna(how='any',axis=0)  ## axis 0 is rowwise

#print(MyTextDF.head())

#print(MyTextDF["Label"])

#print(MyTextDF.iloc[1,1])


# In[152]:


display(MyTextDF)


# In[136]:


# Python program to generate WordCloud 
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in MyTextDF.Text: 
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


# In[137]:


import numpy as np
from PIL import Image
import urllib
import requests
mask = np.array(Image.open(requests.get('http://www.clker.com/cliparts/0/1/2/7/1194985315887015999movie_fudriot_omic.ch_01.svg.med.png', stream=True).raw))

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


# In[130]:


## KEEP THE LABELS!

MyLabel = MyTextDF["Label"]

## Remove the labels from the DF

DF_noLabel= MyTextDF.drop(["Label"], axis=1)  #axis 1 is column

print(DF_noLabel.head())

## Create a list where each element in the list is a row from

## the file/DF

#print(DF_noLabel)

print("length: ", len(DF_noLabel))


# In[58]:


### BUILD the LIST that "content" in CountVectorizer will expect

MyList=[]  #empty list

for i in range(0,len(DF_noLabel)):

    NextText=DF_noLabel.iloc[i,0]  ## what is this??

    ## PRINT TO FIND OUT!

    #print(MyTextDF.iloc[i,1])

    #print("Review #", i, "is: ", NextText, "\n\n")

    #print(type(NextText))

    ## This list is a collection of all the reviews. It will be HUGE

    MyList.append(NextText)



## see what this list looks like....

print(MyList[1:4])


# In[160]:


########## Now we will vectorize!

## CountVectorizer takes input as content

## But- you cannot use "content" unless you know what

## this means and so what the CountVectorizer expects.

## "content" means that you will need a LIST that

## contains all the text. In other words, the first element in

## the LIST is ALL the text from review 1 (in this case)

## the second element in the LIST will be all the text from

## review 2, and so on...

## If you look ABOVE, the for loop BUILDS this LIST.

    ############################################################

MycountVect = CountVectorizer(input="content",
                             stop_words='english')



CV = MycountVect.fit_transform(MyList)



MyColumnNames=MycountVect.get_feature_names()

VectorizedDF_Text=pd.DataFrame(CV.toarray(),columns=MyColumnNames)

## Note - this DF starts at row 0 (not 1)

## My labels start at 1 so I need to shift by 1

print(VectorizedDF_Text)




# In[175]:


custom_stopwords = ["abcs"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

MyTF = TfidfVectorizer(input="content",
                             stop_words=stopwords)



TF = MyTF.fit_transform(MyList)

ColNames=MyTF.get_feature_names()



## OK good - but we want a document topic model A DTM (matrix of counts)

DataFrame_TF=pd.DataFrame(TF.toarray(), columns=ColNames)

MyColumnNames=MyTF.get_feature_names()

TFVectorizedDF_Text=pd.DataFrame(TF.toarray(),columns=MyColumnNames)

## Note - this DF starts at row 0 (not 1)

## My labels start at 1 so I need to shift by 1

print(TFVectorizedDF_Text)


# In[157]:


### Put the labels back

## Make copy

print(MyLabel)

print(type(MyLabel))  


# In[158]:



NEW_Labels = MyLabel.to_frame()   #index to 0

print(type(NEW_Labels))



NEW_Labels.index =NEW_Labels.index-1

print(NEW_Labels)


# In[176]:


LabeledCLEAN_DF=TFVectorizedDF_Text

LabeledCLEAN_DF["LABEL"]=NEW_Labels

print(LabeledCLEAN_DF)


# In[179]:


################ Method 2: Save csv directly --


LabeledCLEAN_DF.to_csv(r'C:/Users/KnudseQ/Desktop/Homework 3/LabeledCLEAN_DF_Movie Data.csv', index = False)


# In[180]:


display(LabeledCLEAN_DF)


# In[184]:


df = LabeledCLEAN_DF
df = df[ ['LABEL'] + [ col for col in df.columns if col != 'LABEL' ] ]


# In[185]:


display(df)


# In[ ]:




