import pickle

import nltk
import numpy as np

from preprocess import tokenize_tweet

class Classifier:

    def __init__(self):

        with open('model.pickle', 'rb') as f:
            self.model = pickle.load(f)

        with open('vocabulary.pickle', 'rb') as f:
            self.vocabulary = pickle.load(f)

        self.tknzr = nltk.tokenize.RegexpTokenizer(r'\w+')
        self.stopwords = set(nltk.corpus.stopwords.words('english'))
        self.labels = ['neutral', 'positive', 'negative']

    def classify(self, tweet):

        features = self.extract_features(tweet)
        x = features.reshape(1, -1)
        y = self.model.predict(x)
        label = self.labels[y[0]]

        return label

    def extract_features(self, tweet):

        tokens = tokenize_tweet(self.tknzr, self.stopwords, tweet)
        features = np.zeros(len(self.vocabulary), dtype=np.bool)
        for i, token in enumerate(self.vocabulary):
            features[i] = (token in tokens)
        return features

def main():

    classifier = Classifier()

    while True:
        tweet = input('tweet > ')
        print(classifier.classify(tweet))

if __name__ == '__main__':
    main()
