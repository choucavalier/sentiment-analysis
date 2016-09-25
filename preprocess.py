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

    # tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)
    tknzr = nltk.tokenize.RegexpTokenizer(r'\w+')

    stopwords = set(nltk.corpus.stopwords.words('english'))

    datasets = [
        'raw_data/train/train_airlines.csv',
        'raw_data/train/train_apple.csv',
        'raw_data/train/train_products.csv',
        'raw_data/test/test_airlines.csv',
        'raw_data/test/test_apple.csv',
        'raw_data/test/test_products.csv',
    ]

    data_tokens = []
    all_tokens = []

    for dataset_path in datasets:
        with open(dataset_path, encoding='utf-8') as raw_file:
            buffer = u''
            for line in raw_file.readlines():
                if len(buffer) > 0:
                    line = buffer + line
                try:
                    tweet, label = line.rsplit(',', 1)
                    buffer = ''
                    label = label.rstrip('\n')
                    tokens = tokenize_tweet(tknzr, stopwords, tweet)
                    data_tokens.append((tokens, label))
                    all_tokens += tokens
                except ValueError:
                    buffer = line

    dist = nltk.FreqDist(all_tokens)

    vocabulary = list([w for w, f in dist.most_common(2000)])

    with open('vocabulary.pickle', 'wb') as f:
        pickle.dump(vocabulary, f)

    data = []

    for tokens, label in data_tokens:
        features = np.zeros(len(vocabulary), dtype=np.bool)
        for i, token in enumerate(vocabulary):
            if token in tokens:
                # print(token, 'in tweet')
                features[i] = True
        data.append((features, label))

    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
