import unittest
import pandas as pd

from ..fix_extract_dataframe import read_json
from ..fix_extract_dataframe import TweetDfExtractor

_, tweet_list = read_json("../covid19.json")

columns = ['created_at', 'source', 'original_text','clean_text', 'sentiment','polarity','subjectivity', 'lang', 'favorite_count', 'retweet_count', 
    'original_author', 'screen_count', 'followers_count','friends_count','possibly_sensitive', 'hashtags', 'user_mentions', 'place', 'place_coord_boundaries']

class TestTweetDfExtractor(unittest.TestCase):

    def setUp(self) -> pd.DataFrame:
        self.df = TweetDfExtractor(tweet_list, columns)

    def test_find_statuses_count(self):
        self.assertEqual(self.df.find_statuses_count(),)