# -*- coding: utf-8 -*-
"""Text Analytics HW 8.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/176S4AUjMZiPjKW_CgcxcQGPtVhL7b_vJ
"""

import requests
import json
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

## GO HERE...GET API KEY...
## https://newsapi.org/
## https://newsapi.org/register
BaseURL="https://newsapi.org/v1/articles"
#BaseURL="https://newsapi.org/v2/top-headlines"

########################################################
## WAY 1  
URLPost = {'apiKey': '1e5f297ff702450fa1762412a92b64eb',
                	'source': 'bbc-news',
                	'pageSize': 85,
                	'sortBy' : 'top',
                	'totalRequests': 75}

response1=requests.get(BaseURL, URLPost)
jsontxt = response1.json()
print(jsontxt)

####################################################

### WAY 2
url = ('https://newsapi.org/v2/everything?'
   	'q=Sports&'
   	'from=2019-11-20&'
   	'sortBy=relevance&'
   	'source=bbc-news&'
   	'pageSize=100&'
   	'1e5f297ff702450fa1762412a92b64eb')

response2 = requests.get(url)
jsontxt2 = response2.json()
print(jsontxt2, "\n")
#####################################################

## Create a new csv file to save the headlines
MyFILE=open("BBCNews.csv","w")
### Place the column names in - write to the first row
WriteThis="Author,Title,Headline\n"
MyFILE.write(WriteThis)
MyFILE.close()

## Open the file for append
MyFILE=open("BBCNews.csv", "a")
for items in jsontxt["articles"]:
	print(items)
         	 
	Author=items["author"]
    
    
	## CLEAN the Title
	##----------------------------------------------------------
	##Replace punctuation with space
	# Accept one or more copies of punctuation    	 
	# plus zero or more copies of a space
	# and replace it with a single space
	Title=items["title"]
	Title=re.sub(r'[,.;@#?!&$\-\']+', ' ', Title, flags=re.IGNORECASE)
	Title=re.sub(r'\ +', ' ', Title, flags=re.IGNORECASE)
	Title=re.sub(r'\"', ' ', Title, flags=re.IGNORECASE)
    
	# and replace it with a single space
	## NOTE: Using the "^" on the inside of the [] means
	## we want to look for any chars NOT a-z or A-Z and replace
	## them with blank. This removes chars that should not be there.
	Title=re.sub(r'[^a-zA-Z]', " ", Title, flags=re.VERBOSE)
	##----------------------------------------------------------
    
	Headline=items["description"]
	Headline=re.sub(r'[,.;@#?!&$\-\']+', ' ', Headline, flags=re.IGNORECASE)
	Headline=re.sub(r'\ +', ' ', Headline, flags=re.IGNORECASE)
	Headline=re.sub(r'\"', ' ', Headline, flags=re.IGNORECASE)
	Headline=re.sub(r'[^a-zA-Z]', " ", Headline, flags=re.VERBOSE)

	#print("Author: ", Author, "\n")
	#print("Title: ", Title, "\n")
	#print("Headline News Item: ", Headline, "\n\n")
    
	WriteThis=Author+ "," + Title + "," + Headline + "\n "
    
	MyFILE.write(WriteThis)
    
## CLOSE THE FILE
MyFILE.close()

## The output looks like this:
##Author:  BBC News

##Title:  Pope Francis addresses violence against women on Colombia visit

##Headline News Item:  Pope Francis calls for respect for
##"strong and influential" women during a five-day trip to Colombia.
##--------------------------------------------------------
#FYI    
#do = jsontxt['articles'][0]["author"]
#print(do)

############### PROCESS THE FILE ######################
## https://stackoverflow.com/questions/21504319/python-3-csv-file-giving-unicodedecodeerror-utf-8-codec-cant-decode-byte-err
## Read to DF
BBC_DF=pd.read_csv("BBCNews.csv", error_bad_lines=False)
print(BBC_DF.head())
BBC_DF.head()

# Commented out IPython magic to ensure Python compatibility.
# Python program to generate WordCloud 

import nltk

import pandas as pd

import sklearn

import re  

from sklearn.feature_extraction.text import CountVectorizer



#Convert a collection of raw documents to a matrix of TF-IDF features.

#Equivalent to CountVectorizer but with tf-idf norm

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud, STOPWORDS 
import nltk
from nltk.corpus import stopwords



from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
# %matplotlib inline
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in BBC_DF.Headline: 
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

# iterating the columns
for col in BBC_DF.columns:
	print(col)
    
print(BBC_DF["Headline"])

### Tokenize and Vectorize the Headlines
## Create the list of headlines
HeadlineLIST=[]
for next in BBC_DF["Headline"]:
	HeadlineLIST.append(next)

print("The headline list is")
print(HeadlineLIST)

import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

#NOTES
##### Installing gensim caused my Spyder IDE no fail and no re-open
## I used two things and did a restart
## 1) in cmd (if PC)  psyder --reset
## 2) in cmd (if PC) conda upgrade qt

######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

data_text_small = BBC_DF[['Headline']]
print(data_text_small)
#data_text['index'] = data_text.index
data_text_small['index'] = data_text_small.index
#print(data_text_small.index)
#print(data_text_small['index'])

#documents = data_text
documents = data_text_small
print(documents)

print("The length of the file - or number of docs is", len(documents))
print(documents[:5])

###################################################
###
### Data Prep and Pre-processing
###
###################################################

import gensim
## IMPORTANT - you must install gensim first ##
## conda install -c anaconda gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer

import numpy as np
np.random.seed(2018)

import nltk
nltk.download('wordnet')
from nltk import PorterStemmer
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

from nltk.tokenize import word_tokenize
from nltk.stem.porter import *

#NOTES
##### Installing gensim caused my Spyder IDE no fail and no re-open
## I used two things and did a restart
## 1) in cmd (if PC)  psyder --reset
## 2) in cmd (if PC) conda upgrade qt

######################################
## function to perform lemmatize and stem preprocessing
############################################################
## Function 1
def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

## Function 2
def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

#Select a document to preview after preprocessing
doc_sample = documents[documents['index'] == 1].values[0][0]
print(doc_sample)
print('original document: ')
words = []
for word in doc_sample.split(' '):
    words.append(word)
print(words)
print('\n\n tokenized and lemmatized document: ')
print(preprocess(doc_sample))

## Preprocess the headline text, saving the results as ‘processed_docs’
processed_docs = documents['Headline'].map(preprocess)
print(processed_docs[:10])

## Create a dictionary from ‘processed_docs’ containing the
## number of times a word appears in the training set.

dictionary = gensim.corpora.Dictionary(processed_docs)

## Take a look ...you can set count to any number of items to see
## break will stop the loop when count gets to your determined value
count = 0
for k, v in dictionary.iteritems():
    print(k, v)
    count += 1
    if count > 5:
        break
   
#print(processed_docs)

## Filter out tokens that appear in
## - - less than 15 documents (absolute number) or
## - - more than 0.5 documents (fraction of total corpus size, not absolute number).
## - - after the above two steps, keep only the first 100000 most frequent tokens
 ############## NOTE - this line of code did not work with my small sample
## as it created blank lists.....      
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

for doc in processed_docs:
    print(doc)

print(dictionary)

## Filter out tokens that appear in
## - - less than 15 documents (absolute number) or
## - - more than 0.5 documents (fraction of total corpus size, not absolute number).
## - - after the above two steps, keep only the first 100000 most frequent tokens
 ############## NOTE - this line of code did not work with my small sample
## as it created blank lists.....      
#dictionary.filter_extremes(no_below=15, no_above=0.5, keep_n=100000)

for doc in processed_docs:
    print(doc)

print(dictionary)

#######################
## For each document we create a dictionary reporting how many
##words and how many times those words appear. Save this to ‘bow_corpus’
##############################################################################
#### bow: Bag Of Words
bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs]
print(bow_corpus[3:5])

#################################################################
### TF-IDF
#################################################################
##Create tf-idf model object using models.TfidfModel on ‘bow_corpus’
## and save it to ‘tfidf’, then apply transformation to the entire
## corpus and call it ‘corpus_tfidf’. Finally we preview TF-IDF
## scores for our first document.

from gensim import corpora, models

tfidf = models.TfidfModel(bow_corpus)
corpus_tfidf = tfidf[bow_corpus]
## pprint is pretty print
from pprint import pprint

for doc in corpus_tfidf:
    pprint(doc)
    ## the break will stop it after the first doc
    break

#############################################################
### Running LDA using Bag of Words
#################################################################
   
#lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=3, id2word=dictionary, passes=1)
   
################################################################
## sklearn
###################################################################3
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
 
NUM_TOPICS = 3



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

### Vectorize
#https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
MyCountV=TfidfVectorizer(input="content", lowercase=True, stop_words = "english")
 
MyDTM = MyCountV.fit_transform(HeadlineLIST)  # create a sparse matrix
print(type(MyDTM))
#vocab is a vocabulary list
vocab = MyCountV.get_feature_names()  # change to a list

MyDTM = MyDTM.toarray()  # convert to a regular array
print(list(vocab)[10:20])
ColumnNames=MyCountV.get_feature_names()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)
print(MyDTM_DF)
MyDTM_DF.head()

from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
#######

#MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
#Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
#ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
#CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
#print(CorpusDF_DH)

######

num_topics = 4

lda_model_DH = LatentDirichletAllocation(n_components=num_topics, max_iter=100, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(MyDTM_DF)


print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)

# Let's see how the first document in the corpus looks like in
## different topic spaces
#print("First headline...")
#print(LDA_DH_Model[0])
#print("Sixth headline...")
#print(LDA_DH_Model[5])

#print(lda_model_DH.components_)


## implement a print function
## REF: https://nlpforhackers.io/topic-modeling/
def print_topics(model, vectorizer, top_n=10):
  for idx, topic in enumerate(model.components_):
        print("Topic:  ", idx)
        print([(vectorizer.get_feature_names()[i], topic[i])
                    	for i in topic.argsort()[:-top_n - 1:-1]])
                    	## gets top n elements in decreasing order

####### call the function above with our model and CountV
print_topics(lda_model_DH, MyCountV)
## Print LDA using print function from above
########## Other Notes ####################
#import pyLDAvis.sklearn as LDAvis
#import pyLDAvis
#import pyLDAvis.gensim
## conda install -c conda-forge pyldavis
#pyLDAvis.enable_notebook() ## not using notebook
#panel = LDAvis.prepare(lda_model_DH, MyDTM_DF, MyCountV, mds='tsne')
#pyLDAvis.show(panel)
#panel = pyLDAvis.gensim.prepare(lda_model_DH, MyDTM, MyCountV, mds='tsne')
#pyLDAvis.show(panel)
##########################################################################

import matplotlib.pyplot as plt
import numpy as np

word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 20

for t in range(num_topics):
	plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
	plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
	plt.xticks([])  # remove x-axis markings ('ticks')
	plt.yticks([]) # remove y-axis markings ('ticks')
	plt.title('Topic #{}'.format(t))
	top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
	top_words_idx = top_words_idx[:num_top_words]
	top_words = vocab_array[top_words_idx]
	top_words_shares = word_topic[top_words_idx, t]
	for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
         plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
             	##fontsize_base*share)

plt.tight_layout()
plt.show()









































"""Alternative LDA w/ Internet Example and News Dataset from course files"""

## =======================================================
## VISUALIZING
## =======================================================  
!pip install pyLDAvis

import pyLDAvis
import pyLDAvis.gensim_models

## =======================================================
## MODELING
## =======================================================
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS

def run_lda(data, num_topics, stop_words):
    regex = "[a-zA-Z]{3,15}"
    cv = CountVectorizer(stop_words = stop_words, token_pattern = regex)
    lda_vec = cv.fit_transform(data)
    lda_columns = cv.get_feature_names()
    corpus = pd.DataFrame(lda_vec.toarray(), columns = lda_columns)
    lda = LatentDirichletAllocation(n_components=num_topics, max_iter=10, 
                                    learning_method='online')
    lda_model = lda.fit_transform(lda_vec)
    print_topics(lda, cv)
    return lda_model, lda, lda_vec, cv, corpus


## =======================================================
## HELPERS
## =======================================================
import numpy as np
np.random.seed(210)

def print_topics(model, vectorizer, top_n=10):
    for idx, topic in enumerate(model.components_):
        print("Topic %d:" % (idx))
        print([(vectorizer.get_feature_names()[i], topic[i])
                        for i in topic.argsort()[:-top_n - 1:-1]])
        

## =======================================================
## VISUALIZING
## =======================================================        
import pyLDAvis.sklearn as LDAvis
import pyLDAvis

def start_vis(lda, lda_vec, cv):
    panel = LDAvis.prepare(lda, lda_vec, cv, mds='tsne')
#     pyLDAvis.show(panel)
    pyLDAvis.save_html(panel, 'HW8_V3_all.html')

from sklearn.feature_extraction import text 
additional_stopwords = [
 '2007',
 '2008',
 'act',
 'american',
 'chairman',
 'committee',
 'congress',
 'country',
 'doc',
 'docno',
 'don',
 'floor',
 'going',
 'government',
 'house',
 'important',
 'just',
 'know',
 'legislation',
 'like',
 'madam',
 'make',
 'members',
 'mr',
 'mrs',
 'ms',
 'need',
 'new',
 'people',
 'president',
 'representatives',
 'say',
 'speaker',
 'state',
 'states',
 'support',
 'text',
 'thank',
 'think',
 'time',
 'today',
 'want',
 'work',
 'year'
]
stop_words = text.ENGLISH_STOP_WORDS.union(additional_stopwords)

import pandas as pd
a_data = pd.read_csv('/content/abcnews_date_text_Kaggle_subset100.csv', error_bad_lines=False);

print(a_data.head())
a_data.head()

# Remove the columns
papers = a_data.drop(columns=['publish_date'], axis=1)
# sample only 100 papers
papers = papers.sample(100)
# Print out the first rows of papers
papers.head()

# Load the regular expression library
import re
# Remove punctuation
papers['paper_text_processed'] = papers['headline_text'].map(lambda x: re.sub('[,\.!?]', '', x))
# Convert the titles to lowercase
papers['paper_text_processed'] = papers['paper_text_processed'].map(lambda x: x.lower())
# Print out the first rows of papers
papers['paper_text_processed'].head()

import gensim
from gensim.utils import simple_preprocess
def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))  # deacc=True removes punctuations
data = papers.paper_text_processed.values.tolist()
data_words = list(sent_to_words(data))
print(data_words[:1][0][:30])

# Build the bigram and trigram models
bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
# Faster way to get a sentence clubbed as a trigram/bigram
bigram_mod = gensim.models.phrases.Phraser(bigram)
trigram_mod = gensim.models.phrases.Phraser(trigram)

# NLTK Stop words
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words.extend(['from', 'subject', 're', 'edu', 'use'])
# Define functions for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(texts):
    return [[word for word in simple_preprocess(str(doc)) if word not in stop_words] for doc in texts]
def make_bigrams(texts):
    return [bigram_mod[doc] for doc in texts]
def make_trigrams(texts):
    return [trigram_mod[bigram_mod[doc]] for doc in texts]
def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return texts_out

import spacy
# Remove Stop Words
data_words_nostops = remove_stopwords(data_words)
# Form Bigrams
data_words_bigrams = make_bigrams(data_words_nostops)
# Initialize spacy 'en' model, keeping only tagger component (for efficiency)
nlp = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
# Do lemmatization keeping only noun, adj, vb, adv
data_lemmatized = lemmatization(data_words_bigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
print(data_lemmatized[:1])

import gensim.corpora as corpora
# Create Dictionary
id2word = corpora.Dictionary(data_lemmatized)
# Create Corpus
texts = data_lemmatized
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]
# View
print(corpus[:1])

# Build LDA model
lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                       id2word=id2word,
                                       num_topics=4, 
                                       random_state=100,
                                       chunksize=100,
                                       passes=10,
                                       per_word_topics=True)

from pprint import pprint
# Print the Keyword in the 10 topics
pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

from gensim.models import CoherenceModel
# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

# supporting function
def compute_coherence_values(corpus, dictionary, k, a, b):
    
    lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=dictionary,
                                           num_topics=k, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=a,
                                           eta=b)
    
    coherence_model_lda = CoherenceModel(model=lda_model, texts=data_lemmatized, dictionary=id2word, coherence='c_v')
    
    return coherence_model_lda.get_coherence()

import numpy as np
import tqdm
grid = {}
grid['Validation_Set'] = {}
# Topics range
min_topics = 2
max_topics = 11
step_size = 1
topics_range = range(min_topics, max_topics, step_size)
# Alpha parameter
alpha = list(np.arange(0.01, 1, 0.3))
alpha.append('symmetric')
alpha.append('asymmetric')
# Beta parameter
beta = list(np.arange(0.01, 1, 0.3))
beta.append('symmetric')
# Validation sets
num_of_docs = len(corpus)
corpus_sets = [# gensim.utils.ClippedCorpus(corpus, num_of_docs*0.25), 
               # gensim.utils.ClippedCorpus(corpus, num_of_docs*0.5), 
               gensim.utils.ClippedCorpus(corpus, int(num_of_docs*0.75)), 
               corpus]
corpus_title = ['75% Corpus', '100% Corpus']
model_results = {'Validation_Set': [],
                 'Topics': [],
                 'Alpha': [],
                 'Beta': [],
                 'Coherence': []
                }
# Can take a long time to run
if 1 == 1:
    pbar = tqdm.tqdm(total=540)
    
    # iterate through validation corpuses
    for i in range(len(corpus_sets)):
        # iterate through number of topics
        for k in topics_range:
            # iterate through alpha values
            for a in alpha:
                # iterare through beta values
                for b in beta:
                    # get the coherence score for the given parameters
                    cv = compute_coherence_values(corpus=corpus_sets[i], dictionary=id2word, 
                                                  k=k, a=a, b=b)
                    # Save the model results
                    model_results['Validation_Set'].append(corpus_title[i])
                    model_results['Topics'].append(k)
                    model_results['Alpha'].append(a)
                    model_results['Beta'].append(b)
                    model_results['Coherence'].append(cv)
                    
                    pbar.update(1)
    pd.DataFrame(model_results).to_csv('lda_tuning_results.csv', index=False)
    pbar.close()

lda_model = gensim.models.LdaMulticore(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=6, 
                                           random_state=100,
                                           chunksize=100,
                                           passes=10,
                                           alpha=0.91,
                                           eta=0.91)

#import pyLDAvis.gensim

import pickle 
import pyLDAvis as LDAvis
import pyLDAvis.gensim_models as gensimvis
# Visualize the topics
pyLDAvis.enable_notebook()
vocab = corpus
term_frequency = id2word
LDAvis_prepared = LDAvis.prepare(lda_model, corpus, id2word)
LDAvis_prepared

# Commented out IPython magic to ensure Python compatibility.
# Python program to generate WordCloud 

import nltk

import pandas as pd

import sklearn

import re  

from sklearn.feature_extraction.text import CountVectorizer



#Convert a collection of raw documents to a matrix of TF-IDF features.

#Equivalent to CountVectorizer but with tf-idf norm

from sklearn.feature_extraction.text import TfidfVectorizer

from wordcloud import WordCloud, STOPWORDS 
import nltk
from nltk.corpus import stopwords



from nltk.tokenize import word_tokenize

from nltk.probability import FreqDist

import matplotlib.pyplot as plt

from nltk.corpus import stopwords
# %matplotlib inline
from matplotlib import pyplot as plt
#
# 

comment_words = '' 
custom_stopwords = ["go", "going"]
stopwords = set(STOPWORDS).union(custom_stopwords) 

# iterate through the csv file 
for val in a_data.headline_text: 
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

MyDTM = MyCountV.fit_transform(a_data.headline_text)  # create a sparse matrix
print(type(MyDTM))
#vocab is a vocabulary list
vocab = MyCountV.get_feature_names()  # change to a list

MyDTM = MyDTM.toarray()  # convert to a regular array
print(list(vocab)[10:20])
ColumnNames=MyCountV.get_feature_names()
MyDTM_DF=pd.DataFrame(MyDTM,columns=ColumnNames)
print(MyDTM_DF)
MyDTM_DF.head()

#MyVectLDA_DH=CountVectorizer(input='filename')
##path="C:\\Users\\profa\\Documents\\Python Scripts\\TextMining\\DATA\\SmallTextDocs"
#Vect_DH = MyVectLDA_DH.fit_transform(ListOfCompleteFiles)
#ColumnNamesLDA_DH=MyVectLDA_DH.get_feature_names()
#CorpusDF_DH=pd.DataFrame(Vect_DH.toarray(),columns=ColumnNamesLDA_DH)
#print(CorpusDF_DH)

######

num_topics = 4

lda_model_DH = LatentDirichletAllocation(n_components=num_topics, max_iter=100, learning_method='online')
#lda_model = LatentDirichletAllocation(n_components=NUM_TOPICS, max_iter=10, learning_method='online')
LDA_DH_Model = lda_model_DH.fit_transform(MyDTM_DF)


print("SIZE: ", LDA_DH_Model.shape)  # (NO_DOCUMENTS, NO_TOPICS)
print_topics(lda_model_DH, MyCountV)

word_topic = np.array(lda_model_DH.components_)
word_topic = word_topic.transpose()

num_top_words = 10
vocab_array = np.asarray(vocab)

#fontsize_base = 70 / np.max(word_topic) # font size for word with largest share in corpus
fontsize_base = 20

for t in range(num_topics):
	plt.subplot(1, num_topics, t + 1)  # plot numbering starts with 1
	plt.ylim(0, num_top_words + 0.5)  # stretch the y-axis to accommodate the words
	plt.xticks([])  # remove x-axis markings ('ticks')
	plt.yticks([]) # remove y-axis markings ('ticks')
	plt.title('Topic #{}'.format(t))
	top_words_idx = np.argsort(word_topic[:,t])[::-1]  # descending order
	top_words_idx = top_words_idx[:num_top_words]
	top_words = vocab_array[top_words_idx]
	top_words_shares = word_topic[top_words_idx, t]
	for i, (word, share) in enumerate(zip(top_words, top_words_shares)):
         plt.text(0.3, num_top_words-i-0.5, word, fontsize=fontsize_base)
             	##fontsize_base*share)

plt.tight_layout()
plt.show()

sm_test = a_data['headline_text'].values[:10]
lda_model, lda, lda_vec, cv, corpus = run_lda(sm_test, 10, stop_words)

# sm_test = a_data['0'].values[:10]
# lda_model, lda, lda_vec, cv, corpus = run_lda(sm_test, 10, stop_words)

lda_model, lda, lda_vec, cv, corpus = run_lda(a_data['headline_text'].values, 4, stop_words)

def start_vis(lda, lda_vec, cv):
    panel = LDAvis.prepare(lda, lda_vec, cv, mds='tsne')
#     pyLDAvis.show(panel)
    pyLDAvis.save_html(panel, 'HW8_V3_all.html')

#!pip install --upgrade pandas==1.2
#!pip install numexpr
#!pip install --upgrade pandas

start_vis(lda, lda_vec, cv)

a_data = pd.read_csv('/content/BBCNews.csv', error_bad_lines=False);

print(a_data.head())

sm_test2 = a_data['Headline'].values[:10]
lda_model, lda, lda_vec, cv, corpus = run_lda(sm_test2, 10, stop_words)

lda_model, lda, lda_vec, cv, corpus = run_lda(a_data['Headline'].values, 4, stop_words)

def start_vis(lda, lda_vec, cv):
    panel = LDAvis.prepare(lda, lda_vec, cv, mds='tsne')
#     pyLDAvis.show(panel)
    pyLDAvis.save_html(panel, 'HW8_all2.html')

start_vis(lda, lda_vec, cv)