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
        
    def test_drop_unwanted_column(self):
        """
        - this function tests the drop unwanted columns
        """
        self.assertLess(len(self.df.columns),len(self.df_cleanser.drop_unwanted_column(self.df).columns))
    
    
    def test_drop_duplicate(self):
        """
        - this function tests the drop duplicates function
        """
        self.assertLess(len(self.df),len(self.df_cleanser.drop_duplicate(self.df)))
    

    def test_convert_to_datetime(self):
        """
        - this function tests the conversion of datetime
        """
        self.assertIsInstance(self.df_cleanser.convert_to_datetime()['created_at'],())
    
    def test_convert_to_numbers(self):
        """
        - this function tests the convert to numeric function
        """
        self.assertIsInstance(self.df_cleanser.convert_to_datetime()['created_at'],())




if __name__ == '__main__':
	unittest.main()

    