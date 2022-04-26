import unittest
import pandas as pd
from clean_tweets_dataframe import TweetCleanser


class TestTweetCleanser(unittest.TestCase):
    """
    - this class tests the TweeetCleanser class
    """


    def setUp(self) -> pd.DataFrame:
        """
        - this function is for the simple setting up of the dataframe
        """
        self.df = pd.read_csv("data/processed_tweet_data.csv")
        self.df_cleanser = TweetCleanser(self.df)
        
    def test_drop_unwanted_column(self, df:pd.DataFrame)->pd.DataFrame:
        """
        - this function tests the drop unwanted columns
        """
        self.assertLess(len(self.df.columns),len(self.df_cleanser.drop_unwanted_column(df).columns))
    
    
    def test_drop_duplicate(self, df:pd.DataFrame)->pd.DataFrame:
        """
        - this function tests the drop duplicates function
        """
        self.assertLess(len(self.df),len(self.df_cleanser.drop_duplicate(self.df)))
    

    