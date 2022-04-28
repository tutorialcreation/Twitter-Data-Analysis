import streamlit as st
import pandas as pd
import numpy as np
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
from extract_dataframe import ExtractTweets
from clean_tweets_dataframe import TweetCleanser
from database import DBOps

#######################
#- Database connections
######################

engine = DBOps(is_online=True)


# setting the title of the dashboard
st.title("Topic Modeling And Sentiment Analysis For Tweets")

st.sidebar.title('Analysis and Modeling of Tweets')



######################
# Retrieving Dataset #
######################

#3.- Read data with pandas



extracted_tweets = ExtractTweets("data/Economic_Twitter_Data.json")
df = pd.read_sql('select * from userData',engine.get_engine())
df.dropna()


#######################
# Data PreProcessing
#####################

# cleaning the dataframe
cleanser = TweetCleanser(df)
# drop unwanted columns
cleanser.drop_unwanted_column(df)
# drop duplicate values from original text
cleanser.drop_duplicate(df)
# convert date data to appropriate datetime
cleanser.convert_to_datetime(df)
# remove non english texts
df_ = cleanser.remove_non_english_tweets(df)

st.sidebar.subheader("Data Overview")

if st.sidebar.checkbox("Display Data"):
    st.write(df_.head(20))


if st.sidebar.checkbox("Show Summary Statistics"):
    st.dataframe(df_.describe())

if st.sidebar.checkbox("Data Types"):
    st.write(df_.info())

user = st.sidebar.selectbox("View specific users data",
set(df['original_author']))
user_df = df_[df_['original_author']==user]
st.write(user_df)

##########################
# Exploratory Data Analysis
############################

st.sidebar.subheader("Expolatory Data Analysis")

column = st.sidebar.selectbox(
    'Which variable would you like to explore',
    (df_.columns)
)

hashtags = df_[df_['hashtags'].map(lambda x: len(x[0])) > 0]
flattened_hash_dataframes = pd.DataFrame(
    [hashtag for hashtags_list in hashtags.hashtags
    for hashtag in eval(hashtags_list)],
    columns=['hashtag'])

df_['all_hashtags'] = flattened_hash_dataframes

top_x = st.sidebar.text_input("Top 'x' tweets",10)

if top_x:
    st.bar_chart(df_['all_hashtags'].value_counts()[:int(top_x)])


# plotting tweets by language
# first of all get the tweets by language
tweets_df = pd.DataFrame(columns = ['original_text','lang'])
tweets_df['text'] = df['original_text'].to_list()
tweets_df['lang'] = df['lang'].to_list()
tweets_according_language = tweets_df['lang'].value_counts()
top_x = st.sidebar.text_input("Top 'x' languages",10)

fig,ax = plt.subplots()
ax.tick_params(axis='x',labelsize=10)
ax.tick_params(axis='y',labelsize=10)
ax.set_xlabel("Languages")
ax.set_ylabel("Frequency")
ax.set_title("The 10 Most Frequent Used Languages In Tweets")
tweets_according_language[:int(top_x)].plot(ax=ax,kind='bar',color='green')


if top_x:
    st.pyplot(fig)