
import nltk
import numpy as np
import string
import load_sent
from textblob import TextBlob
import exp_replace

porter = nltk.PorterStemmer()
sentiments = load_sent.load_sent_word_net()

def dialogueactfeatures(sentence,topicmodeler):
        
    features = {}
    
    gramsfeature(features,sentence)
    sentfeature(features,sentence)
    posfeature(features,sentence)
    capfeature(features,sentence)
    topicfeature(features,sentence,topicmodeler)
    
    return features
    
def gramsfeature(features,sentence):
    sentencereg = exp_replace.replace_reg(sentence)
    
    tokens = nltk.word_tokenize(sentencereg)
    tokens = [porter.stem(t.lower()) for t in tokens] 
    bigrams = nltk.bigrams(tokens)
    bigrams = [tup[0]+' ' +tup[1] for tup in bigrams]
    grams = tokens + bigrams
    
    for t in grams:
        features['contains(%s)' % t] = 1.0
        
def sentfeature(features,sentence):
   
    sentencesentiment = exp_replace.replace_emo(sentence)
    tokens = nltk.word_tokenize(sentencesentiment)
    tokens = [(t.lower()) for t in tokens] 
    
    meansentiment = sentiments.score_sentence(tokens)
    features['Positive sentiment'] = meansentiment[0]
    features['Negative sentiment'] = meansentiment[1]
    features['Sentiment'] = meansentiment[0]-meansentiment[1]
    
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in tokens]).strip())
        features['sentiment'] = blob.sentiment.polarity
        features['subjectivity'] = blob.sentiment.subjectivity
    except:
        features['sentiment'] = 0.0
        features['subjectivity'] = 0.0
    
   
    if len(tokens)==1:
        tokens+=['.']
    f_half = tokens[0:len(tokens)/2]
    s_half = tokens[len(tokens)/2:]
    
    
    mean_sentiment_f = sentiments.score_sentence(f_half)
    features['Positive sentiment 1/2'] = mean_sentiment_f[0]
    features['Negative sentiment 1/2'] = mean_sentiment_f[1]
    features['Sentiment 1/2'] = mean_sentiment_f[0]-mean_sentiment_f[1]
    
    mean_sentiment_s = sentiments.score_sentence(s_half)
    features['Positive sentiment 2/2'] = mean_sentiment_s[0]
    features['Negative sentiment 2/2'] = mean_sentiment_s[1]
    features['Sentiment 2/2'] = mean_sentiment_s[0]-mean_sentiment_s[1]
    
    features['Sentiment contrast 2'] = np.abs(features['Sentiment 1/2']-features['Sentiment 2/2'])

    
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in f_half]).strip())
        features['Blob sentiment 1/2'] = blob.sentiment.polarity
        features['Blob subjectivity 1/2'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment 1/2'] = 0.0
        features['Blob subjectivity 1/2'] = 0.0
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in s_half]).strip())
        features['Blob sentiment 2/2'] = blob.sentiment.polarity
        features['Blob subjectivity 2/2'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment 2/2'] = 0.0
        features['Blob subjectivity 2/2'] = 0.0
        
    features['Blob Sentiment contrast 2'] = np.abs(features['Blob sentiment 1/2']-features['Blob sentiment 2/2'])

    if len(tokens)==2:
        tokens+=['.']
    f_half = tokens[0:len(tokens)/3]
    s_half = tokens[len(tokens)/3:2*len(tokens)/3]
    t_half = tokens[2*len(tokens)/3:]
    
    mean_sentiment_f = sentiments.score_sentence(f_half)
    features['Positive sentiment 1/3'] = mean_sentiment_f[0]
    features['Negative sentiment 1/3'] = mean_sentiment_f[1]
    features['Sentiment 1/3'] = mean_sentiment_f[0]-mean_sentiment_f[1]
    
    mean_sentiment_s = sentiments.score_sentence(s_half)
    features['Positive sentiment 2/3'] = mean_sentiment_s[0]
    features['Negative sentiment 2/3'] = mean_sentiment_s[1]
    features['Sentiment 2/3'] = mean_sentiment_s[0]-mean_sentiment_s[1]
    
    mean_sentiment_t = sentiments.score_sentence(t_half)
    features['Positive sentiment 3/3'] = mean_sentiment_t[0]
    features['Negative sentiment 3/3'] = mean_sentiment_t[1]
    features['Sentiment 3/3'] = mean_sentiment_t[0]-mean_sentiment_t[1]
    
    features['Sentiment contrast 3'] = np.abs(features['Sentiment 1/3']-features['Sentiment 3/3'])
    
    #TextBlob sentiment analysis
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in f_half]).strip())
        features['Blob sentiment 1/3'] = blob.sentiment.polarity
        features['Blob subjectivity 1/3'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment 1/3'] = 0.0
        features['Blob subjectivity 1/3'] = 0.0
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in s_half]).strip())
        features['Blob sentiment 2/3'] = blob.sentiment.polarity
        features['Blob subjectivity 2/3'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment 2/3'] = 0.0
        features['Blob subjectivity 2/3'] = 0.0
    try:
        blob = TextBlob("".join([" "+i if not i.startswith("'") and i not in string.punctuation else i for i in t_half]).strip())
        features['Blob sentiment 3/3'] = blob.sentiment.polarity
        features['Blob subjectivity 3/3'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment 3/3'] = 0.0
        features['Blob subjectivity 3/3'] = 0.0
        
    features['Blob Sentiment contrast 3'] = np.abs(features['Blob sentiment 1/3']-features['Blob sentiment 3/3'])
    
def posfeature(features,sentence):
    
    sentence_pos = exp_replace.replace_emo(sentence)
    tokens = nltk.word_tokenize(sentence_pos)
    tokens = [(t.lower()) for t in tokens] 
    pos_vector = sentiments.posvector(tokens)
    for j in range(len(pos_vector)):
        features['POS' + str(j+1)] = pos_vector[j]
        
def capfeature(features,sentence):
    counter = 0
    threshold = 4
    for j in range(len(sentence)):
        counter+=int(sentence[j].isupper())
    features['Capitalization'] = int(counter>=threshold)
    
def topicfeature(features,sentence,topic_modeler):
    
    topics = topic_modeler.transform(sentence)
    
    for j in range(len(topics)):
        features['Topic :' +str(topics[j][0])] = topics[j][1]
    
    