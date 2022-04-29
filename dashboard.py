from bleach import clean
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
    st.subheader(f"The {top_x} most frequent tweets")
    st.bar_chart(df_['all_hashtags'].value_counts()[:int(top_x)])


# cleaning text
df_['clean_text'] = df_['original_text'].apply(cleanser.clean_text)
df_['clean_text'] = df_['clean_text'].astype(str)
df_['clean_text'] = df_['clean_text'].apply(lambda x:x.lower())
df_['clean_text'] = df_['clean_text'].apply(lambda x: x.translate(str.maketrans(' ', ' ', string.punctuation)))

st.sidebar.text("Fine Tune Word Cloud")
x = st.sidebar.text_input("x",1400)
y = st.sidebar.text_input("y",600)
if x and y:
    fig,ax = plt.subplots()
    ax.imshow(WordCloud(width=1500,height=600,stopwords=STOPWORDS).generate(' '.join(df_['clean_text'].values)))
    ax.axis('off')
    ax.set_title("The most frequent words in the tweets",fontsize=18)
    st.pyplot(fig)


##################
# Topic Modeling
###################
st.sidebar.subheader("Topic Modeling")

sentences = [tweet for tweet in df_['clean_text']]
words = [sentence.split() for sentence in sentences]
word_to_id_dict = corpora.Dictionary(words)
bag_of_words = [word_to_id_dict.doc2bow(tweet) for tweet in words]

num_topics=st.sidebar.text_input("num topics",5)
random_state=st.sidebar.text_input("random_state",100)
update_every=st.sidebar.text_input("update_every",1)
chunksize=st.sidebar.text_input("chunksize",100)
passes=st.sidebar.text_input("passes",10)

lda_model = LdaModel(bag_of_words,
                    id2word=word_to_id_dict,
                    num_topics=int(num_topics),
                    random_state=int(random_state),
                    update_every=int(update_every),
                    chunksize=int(chunksize),
                    passes=int(passes),
                    alpha='auto',
                    per_word_topics=True)

perplexity = lda_model.log_perplexity(bag_of_words)
# perplexity score
st.subheader("Perplexity Score of Your Model")
st.write(perplexity)

coherence_model = CoherenceModel(model=lda_model,
                              texts=words,
                              dictionary=word_to_id_dict,
                              coherence='c_v')
coherence_lda = coherence_model.get_coherence()
# coherence score
st.subheader("Coherence Score of Your Model")
st.write(coherence_lda)


#####################
# sentiment analysis
#####################

st.sidebar.subheader("Sentiment Analysis")

sentiment = st.sidebar.radio(
    "What sentiment would you like to analyze?",
    ('negative', 'positive','neutral')
)