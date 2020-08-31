import os.path
import pandas as pd


def load_tweets():
    r"""
    Loads all datasets provided by the MCC and concatenates them into a single DataFrame.

    After loading once, the concatenated dataset is saved in a highly compressed state. 
    Thereafter, only this one is loaded when the function is executed again.

    Returns:
        :obj:`DataFrame[Series[str]]`: Dataset containing all tweets
    """



    if os.path.isfile("./saved_states/tweet_data_raw.pkl"):
        print("tweet data is already saved. Loading ...")
        all_data_tweets=pd.read_pickle("./saved_states/tweet_data_raw.pkl", compression="bz2")
        print("done")
    else:
        print("tweet data must be loaded first")
        path="./datasets/Twitter_data/Hashtags_Klimapolitik/"
        data_sets = ["#CO2Steuern", "#CO2Abgabe", "#CO2Preis", "#CO2Preise", "#Emissionshandel", "#ETS", "#EUETS", "#Klimaschutzgesetz"]
        

        print("loading #CO2 Steuer tweets")
        with open(path+"#CO2Steuer_tweets.json", encoding="utf-8") as json_file:
            data_tweets = pd.read_json(json_file)
            data_tweets = pd.concat([data_tweets,data_tweets.entities.apply(pd.Series)],axis=1, sort=False)
            data_tweets.drop("entities",axis=1,inplace=True)
            
        all_data_tweets = data_tweets
        for hashtag in data_sets:
            with open(path+hashtag+"_tweets.json", encoding="utf-8") as json_file:
                tweets = pd.read_json(json_file)
                tweets = pd.concat([tweets,tweets.entities.apply(pd.Series)],axis=1, sort=False)
                tweets.drop("entities",axis=1,inplace=True)
                
            all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True, sort=False)

        print("loading Erderwärmungstweets")    
        path="./datasets/Twitter_data/klima_etc_tweets/"
        data_sets = ["globale Erwärmung", "Treibhauseffekt"]
        
        with open(path+"Erderwärmung_tweets.json", encoding="utf-8") as json_file:
            data_tweets = pd.read_json(json_file)
            data_tweets = data_tweets['text']
            
        all_data_tweets = data_tweets
        for hashtag in data_sets:
            with open(path+hashtag+"_tweets.json", encoding="utf-8") as json_file:
                tweets = pd.read_json(json_file)
                tweets = tweets['text']
                
            all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True, sort=False)
        
        
        path="./datasets/Twitter_data/klima_etc_tweets/Klima/"
            
        for i in range(22):
            with open(path+str(i)+"_tweets.json", encoding="utf-8") as json_file:
                tweets = pd.read_json(json_file)
                tweets = tweets['text']
                
            all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True, sort=False)
    
        print("loading Klima tweets")
        path="./datasets/Twitter_data/hashtags_klimaschutz_klimawandel_klimakrise/"
        data_sets = ["#Klimakrise", "#Klimaschutz", "#Klimawandel"]
            
        for hashtag in data_sets:
            with open(path+hashtag+"_tweets.json", encoding="utf-8") as json_file:
                tweets = pd.read_json(json_file)
                tweets = tweets['text']
                
            all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True, sort=False)
        
        print("loading german parliament tweets")
        path="./datasets/Twitter_data/tweets_by_german_parliamentarians/"
        data_sets = ["german_parliamentarians_mdb"]
            
        for hashtag in data_sets:
            with open(path+hashtag+"_tweets.json", encoding="utf-8") as json_file:
                tweets = pd.read_json(json_file)
                tweets = tweets['text']
                
            all_data_tweets = pd.concat([all_data_tweets, tweets], ignore_index=True, sort=False)
            
        dirName="./saved_states"
        if not os.path.exists(dirName):
            os.mkdir(dirName)
        
        all_data_tweets.to_pickle("./saved_states/tweet_data_raw.pkl",compression="bz2")
        
        print("done")
        
    return all_data_tweets
        
        