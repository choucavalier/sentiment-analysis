'''Preprocessing module

This module:

1.  creates a vocabulary from a dataset

    the vocabulary is the set of the most common tokens found in the corpus

2.  creates a vector of features for each datum

    the vector contains boolean indicating, for each token in the vocabulary
    if the token is contained in the datum

Both the vocabulary and extracted features are persisted with pickle
'''
import pickle

import numpy as np
import nltk

def tokenize_tweet(tknzr, stopwords, tweet: str):
    tweet = tweet.replace('\n', '').strip('"').rstrip('"')
    tweet = tweet.lower()
    tokens = set(tknzr.tokenize(tweet)) - stopwords
    for token in list(tokens):
        if token.startswith('http'):
            tokens.remove(token)
    return tokens

def main():

    # word tokenizer
    tknzr = nltk.tokenize.RegexpTokenizer(r'\w+')
    # english stopwords that shouldn't be part of the vocabulary
    stopwords = set(nltk.corpus.stopwords.words('english'))

    datasets = [
        'raw_data/train/train_airlines.csv',
        'raw_data/train/train_apple.csv',
        'raw_data/train/train_products.csv',
        'raw_data/test/test_airlines.csv',
        'raw_data/test/test_apple.csv',
        'raw_data/test/test_products.csv',
    ]

    # all filtered tokens appearing in the dataset (with repetition)
    all_tokens = []
    # list of (tokens, label) for each tweet
    data_as_tokens = []

    for dataset_path in datasets:
        with open(dataset_path, encoding='utf-8') as raw_file:
            # using a buffer because the data sometimes is on several lines
            buffer = ''
            for line in raw_file.readlines():
                # if the buffer is not empty, prepend the line with it
                if len(buffer) > 0:
                    line = buffer + line
                try:
                    # attempt to split the tweet and label
                    tweet, label = line.rsplit(',', 1)
                    buffer = ''
                    label = label.rstrip('\n')
                    tokens = tokenize_tweet(tknzr, stopwords, tweet)
                    data_as_tokens.append((tokens, label))
                    all_tokens += tokens
                # raise when a multiline tweet in encountered
                except ValueError:
                    buffer = line

    # calculate frequency of each token in the corpus
    dist = nltk.FreqDist(all_tokens)

    # most common tokens in the corpus
    vocabulary = list([w for w, f in dist.most_common(2000)])

    # persist vocabulary (feature set)
    with open('vocabulary.pickle', 'wb') as f:
        pickle.dump(vocabulary, f)

    # (features, label) for each datum
    data = []

    # extract features for each datum
    for tokens, label in data_as_tokens:
        # initialize a numpy array the size of the vocabulary
        features = np.zeros(len(vocabulary), dtype=np.bool)
        for i, token in enumerate(vocabulary):
            features[i] = (token in tokens)
        data.append((features, label))

    # persist preprocessed dataset ((features, label))
    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
