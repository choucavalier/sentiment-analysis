'''Preprocess the data and generate a vocabulary

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
from gensim.models import doc2vec

VOCABULARY_SIZE = 2000

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

    print('... calculating document vectors')

    # list of (tokens, label) for each tweet
    data_as_tagged_documents = []

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
                    label = label.rstrip('\n')
                    if label not in set(['negative', 'neutral', 'positive']):
                        buffer += line
                        continue
                    buffer = ''
                    tokens = tokenize_tweet(tknzr, stopwords, tweet)
                    tagged_document = doc2vec.TaggedDocument(words=tokens,
                                                             tags=[label])
                    data_as_tagged_documents.append(tagged_document)
                # raise when a multiline tweet in encountered
                except ValueError:
                    buffer += line

    model = doc2vec.Doc2Vec(data_as_tagged_documents, size=100, min_count=10,
                            workers=16)

    model.save('model.d2v')

if __name__ == '__main__':
    main()
