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
from sqlalchemy import types, create_engine
import pymysql

#######################
#- Database connections
######################

try:
    conn = create_engine('mysql+pymysql://user:pass@IP/database_name')
    print("MySQL Connection Sucessfull!!!!!!!!!!!")
except Exception as err:
	print("MySQL Connection Failed !!!!!!!!!!!")
	print(err)

# setting the title of the dashboard
st.title("Topic Modeling And Sentiment Analysis For Tweets")

st.sidebar.title('Analysis and Modeling of Tweets')



######################
# Retrieving Dataset #
######################

extracted_tweets = ExtractTweets("data/Economic_Twitter_Data.json")
df = extracted_tweets.get_tweet_df(save=False)
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


##########################
# Exploratory Data Analysis
############################

st.sidebar.subheader("Expolatory Data Analysis")

column = st.sidebar.selectbox(
    'Which variable would you like to explore',
    (df_.columns)
)
