import pickle

import nltk
import numpy as np
from gensim.models import doc2vec

from preprocess import tokenize_tweet

def extract_features(tknzr, stopwords, d2v_model, tweet):
    tokens = tokenize_tweet(tknzr, stopwords, tweet)
    vect = d2v_model.infer_vector(tokens)
    features = np.zeros(len(vect), dtype=np.float)
    for i in range(len(vect)):
        features[i] = vect[i]
    return features

def main():

    d2v_model = doc2vec.Doc2Vec.load('model.d2v')

    with open('model.pickle', 'rb') as f:
        model = pickle.load(f)

    tknzr = nltk.tokenize.RegexpTokenizer(r'\w+')
    stopwords = set(nltk.corpus.stopwords.words('english'))
    labels = ['neutral', 'positive', 'negative']

    while True:
        tweet = input('tweet > ')
        features = extract_features(tknzr, stopwords, d2v_model, tweet)
        x = features.reshape(1, -1)
        y = model.predict(x)
        label = labels[y[0]]
        print(label)

if __name__ == '__main__':
    main()
