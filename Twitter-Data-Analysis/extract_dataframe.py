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

    
    def read_json(json_file: str)->list:
        """
        - this function returns json data in
        the form of a list of dictionaries
        """
        data = []
        for tweets in open(json_file,'r'):
            data.append(json.loads(tweets))
        return data

    def find_statuses_count(self)->list:
        """
        - this function returns the status counts
        """
        return list(map(lambda tweet: tweet['user']['statuses_count'], self.tweets))