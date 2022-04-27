#!/usr/bin/env python
# coding: utf-8

# Topic Modelling and Sentiment Analysis

# In[3]:


import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import STOPWORDS,WordCloud
from gensim import corpora
import pandas as pd
import statistics
import string
import os
import re


# In[4]:


# inbuilt modules
from extract_dataframe import ExtractTweets
from clean_tweets_dataframe import TweetCleanser


# In[5]:


# dataframe from extracted tweets
extracted_tweets = ExtractTweets("data/Economic_Twitter_Data.json")
df = extracted_tweets.get_tweet_df(save=False)


# Preprocessing Tasks

# In[15]:


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


# In[16]:


df_.head()


# In[18]:


df_.info()


# In[19]:


# checking for any missing values from the data
missing_values = df_.isnull().sum().sum()


# In[20]:


missing_values


# In[24]:


# check the columns that have values
columns_with_null_values = df_.columns[df_.isnull().any()]


# In[25]:


columns_with_null_values


# In[31]:


# univariate analysis on hashtags
def get_hashtags(tweet):
    '''This function will extract hashtags'''
    return re.findall('(#[A-Za-z]+[A-Za-z0-9-_]+)', tweet)


# In[34]:


# apply the method to the text
df_['hashtags'] = df_['original_text'].apply(get_hashtags)


# In[41]:


# we can take the rows that have valid hashtags
hashtags = df_[df_['hashtags'].map(lambda x: len(x)) > 0]


# In[44]:


# how many records have hashtags
len(hashtags)


# In[45]:


# we can flatten the hashtags
flattened_hash_dataframes = pd.DataFrame(
    [hashtag for hashtags_list in hashtags.hashtags
    for hashtag in hashtags_list],
    columns=['hashtag'])


# In[47]:


# we can add it to the main dataframe
df_['all_hashtags'] = flattened_hash_dataframes


# In[48]:


# we can plot the top 10 hashtags
df_['all_hashtags'].value_counts()[:10].plot(kind='bar')


# In[ ]:


# plotting tweets by language


# In[ ]:




