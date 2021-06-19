import json
import pandas as pd
from textblob import TextBlob

def read_json(json_file: str)->list:
    """
    json file reader to open and read json files into a list
    Args:
    -----
    json_file: str - path of a json file
    
    Returns
    -------
    length of the json file and a list of json
    """
    
    tweets_data = []
    for tweets in open(json_file,'r'):
        tweets_data.append(json.loads(tweets))
    
    
    return len(tweets_data), tweets_data

class Tweet_df:
    """
    this function will parse tweets json into a pandas dataframe
    
    Return
    ------
    dataframe
    """
    def __init__(self, tweets_list, columns):
        
        self.tweets_list = tweets_list
        self.columns = columns

    # an example function
    def find_statuses_count(self)->list:
        statuses_count 
        
    def find_full_text(self)->list:
        text = 
       
    
    def find_sentiments(self, text)->list:
        
        return polarity, self.subjectivity

    def find_created_time(self)->list:
       
        return created_at

    def find_source(self)->list:
        source = 

        return source

    def find_screen_name(self)->list:
        screen_name = 

    def find_followers_count(self)->list:
        followers_count = 

    def find_friends_count(self)->list:
        friends_count = 

    def is_sensitive(self)->list:
        try:
            is_sensitive = [x['possibly_sensitive'] for x in self.tweets_list]
        except KeyError:
            is_sensitive = None

        return is_sensitive

    def find_favourite_count(self)->list:
        
    
    def find_retweet_count(self)->list:
        retweet_count = 

    def find_hashtags(self)->list:
        hashtags =

    def find_mentions(self)->list:
        mentions = 

    def find_place(self)->list:
        try:
            xyz = self.tweets_list['place']['bounding_box']['coordinates']
            coordinates = [coord for loc in xyz for coord in loc]
        except TypeError:
            coordinates = None
        
        return coordinates

    def find_location(self)->list:
        try:
            location = self.tweets_list['user']['location']
        except TypeError:
            location = ''
        
        return location

    
        
        
   
        
        
        
if __name__ == "__main__":
    # required column to be generated you should be creative and add more features
    columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']
    _, tweet_list = read_json("../covid19.json")
    tweet = Tweet_df(tweet_list, columns=columns)

    # use all defined functions to generate a dataframe with the specified columns above

    