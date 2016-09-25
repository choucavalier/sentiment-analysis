import pickle

import nltk
import numpy as np

from preprocess import tokenize_tweet

def extract_features(tknzr, stopwords, vocabulary, tweet):
    tokens = tokenize_tweet(tknzr, stopwords, tweet)
    features = np.zeros(len(vocabulary), dtype=np.bool)
    for i, token in enumerate(vocabulary):
        features[i] = (token in tokens)
    return features

def main():

    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    with open('vocabulary.pickle', 'rb') as f:
        vocabulary = pickle.load(f)

    tknzr = nltk.tokenize.RegexpTokenizer(r'\w+')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    labels = ['neutral', 'positive', 'negative']

    while True:
        tweet = input('tweet > ')
        features = extract_features(tknzr, stopwords, vocabulary, tweet)
        x = features.reshape(1, -1)
        y = model.predict(x)
        label = labels[y[0]]
        print(label)

if __name__ == '__main__':
    main()
