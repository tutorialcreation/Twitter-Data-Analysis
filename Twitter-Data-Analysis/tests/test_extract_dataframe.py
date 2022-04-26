import unittest
import pandas as pd
import sys, os
 
sys.path.append(os.path.abspath(os.path.join('../..')))

from extract_dataframe import ExtractTweets



columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']


class TestExtractTweeets(unittest.TestCase):
    """
    - A class that tests the Extracts Tweet functions
	"""


    def setUp(self) -> pd.DataFrame:
        self.tweets = ExtractTweets("/home/martin/Documents/Dscience/Economic_Twitter_Data.json")
        self.df = self.tweets.get_tweet_df(save=False)


    
if __name__ == '__main__':
	unittest.main()

    