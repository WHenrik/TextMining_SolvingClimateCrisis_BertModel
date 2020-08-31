import pandas as pd
import requests
import tweepy


def load_labelled_tweets(consumer_key, consumer_secret):
    r"""
    Loads the datasets provided by the MCC labels them.

        Args:
            consumer_key (:obj:`str`):

            consumer_secret (:obj:`str`):

    Returns:
        :obj:`DataFrame[Series[str]]`: Dataset containing all labelled tweets
    """

    path="./Twitter_data/Hashtags_Klimapolitik/"
    data_sets = ["#CO2Steuern", "#CO2Abgabe", "#CO2Preis", "#CO2Preise", "#Emissionshandel", "#ETS", "#EUETS", "#Klimaschutzgesetz"]


    #entities wurden in mehrere Spalten (hashtags,...) entpackt, ggf. müssen diese auch nochmal entpackt werden
    with open(path+"#CO2Steuer_tweets.json", encoding="utf-8") as json_file:
        data_tweets = pd.read_json(json_file)
        data_tweets= pd.concat([data_tweets,data_tweets.entities.apply(pd.Series)],axis=1)
        data_tweets.drop("fields",axis=1,inplace=True)

    with open(path+"#CO2Steuer_users.json", encoding="utf-8") as json_file:
        data_user= pd.read_json(json_file,orient="none")
        data_user= pd.concat([data_user,data_user.fields.apply(pd.Series)],axis=1)
        data_user.drop("fields",axis=1,inplace=True)

    all_data_tweets = data_tweets
    for hashtag in data_sets:
        with open(path+hashtag+"_tweets.json", encoding="utf-8") as json_file:
            tweets = pd.read_json(json_file)
            tweets= pd.concat([tweets,tweets.entities.apply(pd.Series)],axis=1)
            tweets.drop("entities",axis=1,inplace=True)


        tweets['dataset'] = hashtag
        all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True)


    for hashtag in data_sets:
        with open(path+hashtag+"_users.json", encoding="utf-8") as json_file:
                users = pd.read_json(json_file)
                users= pd.concat([users,users.fields.apply(pd.Series)],axis=1)
                users.drop("fields",axis=1,inplace=True)
        data_user = pd.concat([data_user, users], ignore_index=True)

    #get the lists of the german Bundestag
    #these are the ID's of the party-accounts and the Bundestag
    politicians_ids = [21107582, 14553288, 15902865, 3088296873, 46085533, 26458162, 347792540, 39475170, 844081278, 14341194, 234343491]
    
    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    api = tweepy.API(auth)
    
    for mm in tweepy.Cursor(api.list_members, list_id=912241909002833921).items():
        politicians_ids.append(mm._json['id'])
        
    for mm in tweepy.Cursor(api.list_members, list_id=85872567).items():
        politicians_ids.append(mm._json['id'])

    for mm in tweepy.Cursor(api.list_members, list_id=124976628).items():
        politicians_ids.append(mm._json['id'])
    
    for mm in tweepy.Cursor(api.list_members, list_id=195759036).items():
        politicians_ids.append(mm._json['id'])

    for mm in tweepy.Cursor(api.list_members, list_id=124976541).items():
        politicians_ids.append(mm._json['id'])
    
    for mm in tweepy.Cursor(api.list_members, list_id=124976330).items():
        politicians_ids.append(mm._json['id'])

    #get the lists news agencies
    news_ids = [40227292]
    
    #business media
    for mm in tweepy.Cursor(api.list_members, list_id=125009134).items():
        news_ids.append(mm._json['id'])

    #finance and tax
    for mm in tweepy.Cursor(api.list_members, list_id=90015182).items():
        news_ids.append(mm._json['id'])

    #editors-in-Chief
    for mm in tweepy.Cursor(api.list_members, list_id=107078612).items():
        news_ids.append(mm._json['id'])

    #bruessel
    for mm in tweepy.Cursor(api.list_members, list_id=89936803).items():
        news_ids.append(mm._json['id'])

    #international 
    for mm in tweepy.Cursor(api.list_members, list_id=85872588).items():
        news_ids.append(mm._json['id'])

    #dpa emloyees
    for mm in tweepy.Cursor(api.list_members, list_id=85920406).items():
        news_ids.append(mm._json['id'])

    # get a list of ids from companies and industry representatives
    industry_ids = []
    
    #industry
    for mm in tweepy.Cursor(api.list_members, list_id=87993388).items():
        industry_ids.append(mm._json['id'])

    data_user['label'] = None

    #apply all labels
    industry_descriptions = ["Impr.", "Impressum", "CEO", "COO", "CTO", "CMO", "CFO", "CPO", "offizieller Account", "offizieller Acount", "offizielle Seite"]
    industry_ids_all = []
    ids = []
    labels = []
    
    for index, user in data_user.iterrows():
        ids.append(user['pk'])
        if user['pk'] in industry_ids:
            labels.append('industry')
            industry_ids_all.append(user['pk'])
        elif user['pk'] in news_ids:
            labels.append('news')
        elif user['pk'] in politicians_ids:
            labels.append('politician')
        elif user['label'] == None and any(map(user['description'].__contains__, industry_descriptions)):
            labels.append('industry')
            industry_ids_all.append(user['pk'])
        else:
            labels.append("None")

    # create new df with labels and user ids
    data = {'pk':  ids,'label': labels}
    
    labelled_users = pd.DataFrame (data, columns = ['pk','label'])
    labelled_users = labelled_users.drop_duplicates()

    tweet_texts = []
    tweet_labels = []
    for index, tweet in all_data_tweets.iterrows():
        tweet_texts.append(tweet['text'])
        if tweet['author'] in industry_ids_all:
            tweet_labels.append('industry')
        elif tweet['author'] in news_ids:
            tweet_labels.append('news')
        elif tweet['author'] in politicians_ids:
            tweet_labels.append('politician')
        else:
            tweet_labels.append("None")

    data = {'text':  tweet_texts,'label': tweet_labels}
    labelled_tweets = pd.DataFrame (data, columns = ['text','label'])
    labelled_tweets = labelled_tweets.drop_duplicates()

    #save
    labelled_tweets.to_pickle("labelled_tweets_text.pkl",compression="bz2")
    print("Labelled tweets are saved in ./saved_states/labelled_tweets_text.pkl")
        
    return labelled_tweets
        
        