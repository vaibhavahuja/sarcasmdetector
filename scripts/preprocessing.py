import re
import io

def preprocess(tweet):
    final = []
    if('http' not in tweet and r'\u' not in tweet):
        removingNonAlphabets = re.compile('[^a-zA-Z 0-9]')
        hashtags = re.compile(r'#\w*\s?')
        tags = re.compile(r'@\w*\s?')
        tweet = re.sub(hashtags,'',tweet)
        tweet = re.sub(tags,'', tweet)
        tweet = re.sub(removingNonAlphabets,'',tweet)
        final = [tweet,len(tweet.split())]
    else:
        tweet=""
        final = [tweet,len(tweet.split())]
    return final
