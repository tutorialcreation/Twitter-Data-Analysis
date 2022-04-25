import json

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
        - this function finds the full text of the 
        scope on
        """
        return list(map(lambda tweet: tweet['text'], self.tweets))
        