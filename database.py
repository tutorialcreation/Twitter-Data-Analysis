from sqlalchemy import types, create_engine
import pymysql
import pandas as pd
from extract_dataframe import ExtractTweets
from clean_tweets_dataframe import TweetCleanser
from sqlite3 import Error
import csv

class DBOps:
    """
    What this script does:
    - inserts data from json into sqlite (online)
    - inserts data from json into mysql 
    """


    def __init__(self,is_online=True):
        if is_online:
            try:
                self.conn = create_engine('sqlite:///twitter_sqlite.db') # ensure this is the correct path for the sqlite file. 
                print("SQLITE Connection Sucessfull!!!!!!!!!!!")
            except Exception as err:
                print("SQLITE Connection Failed !!!!!!!!!!!")
                print(err)
        else:
            try:
                self.conn = create_engine('mysql+pymysql://root:luther1996-@localhost/twitter')
                print("MySQL Connection Sucessfull!!!!!!!!!!!")
            except Exception as err:
                print("MySQL Connection Failed !!!!!!!!!!!")
                print(err)
        


    def get_engine(self):
        """
        - this function simply returns the connection
        """
        return self.conn

    def get_df(self):
        """
        - this function returns the data
        to be inserted into the sql table
        """
        extracted_tweets = ExtractTweets("data/Economic_Twitter_Data.json")
        df = extracted_tweets.get_tweet_df(save=False)
        df.reset_index(drop=True, inplace=True)
        df.dropna()
        cleanser = TweetCleanser(df)
        cleanser.drop_unwanted_column(df)
        cleanser.drop_duplicate(df)
        cleanser.convert_to_datetime(df)
        df_ = cleanser.remove_non_english_tweets(df)
        df_['hashtags'] = df_['original_text'].apply(cleanser.get_hashtags).astype(str)
        return df_

    
    def execute_from_script(self,sql_script):
        """
        - this function executes commands
        that come streaming in from sql_scripts
        """
        try:
            sql_file = open(sql_script)
            sql_ = sql_file.read()
            sql_file.close()


            sql_commands = sql_.split(";")
            for command in sql_commands:
                if command:
                    self.conn.execute(command)
            print("Successfully created table")
        except Error as e:
            print(e)
        return
    

    def insert_update_data(self,table):
        """
        - this function pushes data into the table
        """
        df = self.get_df()
        df.to_sql(table, con=self.conn, if_exists='replace')
        print("Successfully pushed the data into the database")
        return 
    
if __name__ == "__main__":
    print("Test DBOpsfile")