import regex as re
import os.path
import pandas as pd


def preprocess_empty_and_retweets(tweet_text):
    if tweet_text is None or tweet_text[:2] == "RT" :
        tweet_text=None
    return tweet_text

def drop_out_empty_and_retweets(tweet_text):
    tweet_text=tweet_text[tweet_text.notnull()]
    return tweet_text

def clean_my_data(tweet_text):
    tweet_text= re.sub(r'https?:\/\/.*[\r\n]*', '', tweet_text, flags=re.MULTILINE)
    tweet_text= re.sub(r'pic.twitter\/\/.*[\r\n]*', '', tweet_text, flags=re.MULTILINE)
    tweet_text= re.sub(r"(?:\@)\S+", "", tweet_text)
    tweet_text = re.sub(r"[:;8xX=][-]?[DPCcOo3p)(]", "", tweet_text)
    tweet_text = re.sub(r'\W +', ' ', tweet_text)
    tweet_text= re.sub(r'http\S+', '', tweet_text)
    return tweet_text


def preprocess_tweets(tweet_text):  
    r"""
    Cleans all tweets in the dataset

    After preprocessing once, the cleaned dataset is saved in a highly compressed state. 
    Thereafter, only this one is loaded when the function is executed again.

    Returns:
        :obj:`DataFrame[Series[str]]`: Dataset containing all cleaned tweets
    """


    if os.path.isfile("./saved_states/tweet_data_preprocessed.pkl"):
        print("tweets are already preprocessed. Loading ...")
        tweet_text=pd.read_pickle("./saved_states/tweet_data_preprocessed.pkl", compression="bz2")
        print("done")
    else:
        print("tweet data has to be preprocessed first")
        tweet_text.apply(lambda x: preprocess_empty_and_retweets(x))
        tweet_text=drop_out_empty_and_retweets(tweet_text)
        tweet_text=tweet_text.apply(lambda x: clean_my_data(x))
        tweet_text.reset_index(drop=True,inplace=True)
        tweet_text.to_pickle("./saved_states/tweet_data_preprocessed.pkl",compression="bz2")
        print("done")

    return tweet_text