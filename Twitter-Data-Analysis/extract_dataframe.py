import json
from textblob import TextBlob

class ExtractTweets:
    """
    - class for extracting tweets
    """
    def __init__(self,json_file: str)->list:
        data = []
        for tweets in open(json_file,'r'):
            data.append(json.loads(tweets))
        self.tweets = data

    def find_statuses_count(self)->list:
        """
        - this function returns the status counts
        """
        return list(map(lambda tweet: tweet['user']['statuses_count'], self.tweets))
    
    def find_full_text(self)->list:
        """
        - this function extracts the full texts of
        the tweets
        """
        return list(map(lambda tweet: tweet['text'], self.tweets))
        

    def find_sentiments(self, data:list)->list:
        """
        - this function finds sentiments from the dataset
        """
        get_polarity = [TextBlob(text).sentiment.polarity for text in data]
        get_subjectivity = [TextBlob(text).sentiment.subjectivity for text in data]
        return get_polarity, get_subjectivity
