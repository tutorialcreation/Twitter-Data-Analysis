#!/usr/bin/env python
# coding: utf-8

# # Topic Modelling and Sentiment Analysis

# In[68]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS,WordCloud
from gensim import corpora
from gensim.models.ldamodel import LdaModel,CoherenceModel
from pprint import pprint
import pandas as pd
import statistics
import string
import os
import re
import pyLDAvis.gensim_models as gensimvis
import pickle 
import pyLDAvis
from random import shuffle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.linear_model import SGDClassifier
from sklearn.svm import LinearSVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import numpy as np 
from joblib import dump, load 
from scipy.sparse import save_npz, load_npz 
from scipy.stats import uniform
from scipy.sparse import csr_matrix


# In[35]:


# inbuilt modules
from extract_dataframe import ExtractTweets
from clean_tweets_dataframe import TweetCleanser


# ## Data Preprocessing

# In[3]:


# dataframe from extracted tweets
extracted_tweets = ExtractTweets("data/Economic_Twitter_Data.json")
df = extracted_tweets.get_tweet_df(save=False)
df.dropna()


# Preprocessing Tasks

# In[4]:


# clean the dataframe
cleanser = TweetCleanser(df)
# drop unwanted columns
cleanser.drop_unwanted_column(df)
# drop duplicate values from original text
cleanser.drop_duplicate(df)
# convert date data to appropriate datetime
cleanser.convert_to_datetime(df)
# remove non english texts
df_ = cleanser.remove_non_english_tweets(df)


# In[36]:


df_.head()


# In[6]:


df_.info()


# In[7]:


# checking for any missing values from the data
missing_values = df_.isnull().sum().sum()


# In[8]:


missing_values


# In[9]:


# check the columns that have values
columns_with_null_values = df_.columns[df_.isnull().any()]


# In[10]:


columns_with_null_values


# ## EDA (Expolaratory Data Analysis)

# In[11]:


# univariate analysis on hashtags
def get_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)


# In[12]:


# apply the method to the text
df_['hashtags'] = df_['original_text'].apply(get_hashtags)


# In[13]:


# we can take the rows that have valid hashtags
hashtags = df_[df_['hashtags'].map(lambda x: len(x)) > 0]


# In[14]:


# how many records have hashtags
len(hashtags)


# In[15]:


# we can flatten the hashtags
flattened_hash_dataframes = pd.DataFrame(
    [hashtag for hashtags_list in hashtags.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


# In[16]:


# we can add it to the main dataframe
df_['all_hashtags'] = flattened_hash_dataframes


# In[17]:


# we can plot the top 10 hashtags
df_['all_hashtags'].value_counts()[:10].plot(kind='bar',color='purple')


# In[18]:


# plotting tweets by language
# first of all get the tweets by language
tweets_df = pd.DataFrame(columns = ['original_text','lang'])
tweets_df['text'] = df['original_text'].to_list()
tweets_df['lang'] = df['lang'].to_list()
tweets_according_language = tweets_df['lang'].value_counts()


# In[19]:


# we can plot the most frequent language
fig,ax = plt.subplots()
ax.tick_params(axis='x',labelsize=10)
ax.tick_params(axis='y',labelsize=10)
ax.set_xlabel("Languages")
ax.set_ylabel("Frequency")
ax.set_title("The 10 Most Frequent Used Languages In Tweets")
tweets_according_language[:10].plot(ax=ax,kind='bar',color='green')


# In[20]:


import enchant


# In[21]:


en_us = enchant.Dict("en_US")


# In[22]:


# text processing
def clean_text(tweet):
    """this function cleans the original text"""
    return ' '.join(w for w in tweet.split() if en_us.check(w))


# In[23]:


# apply the method to the original text
df_['clean_text'] = df_['original_text'].apply(clean_text)


# In[24]:


# text processing
df_['clean_text'] = df_['clean_text'].astype(str)
df_['clean_text'] = df_['clean_text'].apply(lambda x:x.lower())
df_['clean_text'] = df_['clean_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))


# In[25]:


# plot the texts in a wordcloud
plt.figure(figsize=(30,10))
plt.imshow(WordCloud(width=1500,height=600,stopwords=STOPWORDS).generate(' '.join(df_['clean_text'].values)))
plt.axis('off')
plt.title("The most frequent words in the tweets",fontsize=18)
plt.show()


# ## Topic Modelling

# In[26]:


# create a bag of words
sentences = [tweet for tweet in df_['clean_text']]
words = [sentence.split() for sentence in sentences]
word_to_id_dict = corpora.Dictionary(words)
bag_of_words = [word_to_id_dict.doc2bow(tweet) for tweet in words]


# In[27]:


# check bag of words
bag_of_words[1]


# In[28]:


# building an lda model
lda_model = LdaModel(bag_of_words,
                    id2word=word_to_id_dict,
                    num_topics=5,
                    random_state=100,
                    update_every=1,
                    chunksize=100,
                    passes=10,
                    alpha='auto',
                    per_word_topics=True)


# In[29]:


pprint(lda_model.show_topics(formatted=False))


# In[30]:


# perplexity computation
perplexity = lda_model.log_perplexity(bag_of_words)
perplexity


# In[31]:


# coherence score
coherence_model = CoherenceModel(model=lda_model,
                              texts=words,
                              dictionary=word_to_id_dict,
                              coherence='c_v')
coherence_lda = coherence_model.get_coherence()
coherence_lda


# In[32]:


# explore intertopic distances
pyLDAvis.enable_notebook()
prepared = gensimvis.prepare(lda_model,bag_of_words,word_to_id_dict)
prepared


# ## Sentiment Analysis

# In[55]:


# create sentiment column
conditions = [df_.polarity < 0 , df_.polarity > 0,df_.polarity==0]
choices = ['negative', 'positive','neutral']
df_['sentiment'] = np.select(conditions, choices, default='zero')
y = df_['sentiment']


# In[42]:


# creating a tf-idf vector
vector = TfidfVectorizer(max_features=2000,min_df=6,max_df=0.5,stop_words=STOPWORDS)
x = vector.fit_transform(df_['clean_text'])


# In[58]:


# create the test and split data
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3)


# In[61]:


# apply a model
linear_model = LinearSVC()
linear_model.fit(x_train,y_train)


# In[65]:


# predict results
y_predict = linear_model.predict(x_test)
print(classification_report(y_test,y_predict))


# In[105]:


# apply model 2
kmeans = KMeans(init="random", n_clusters=3, n_init=10, max_iter=300, random_state=42 )
kmeans.fit(x_train)
labels = kmeans.fit_predict(x_test)


# In[101]:


# plot our kmeans
sns.scatterplot(data=y, x=y_test, y=labels, hue=kmeans.labels_)
plt.show()


# In[ ]:




