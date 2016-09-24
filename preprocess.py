import pickle

import nltk

def main():

    tknzr = nltk.tokenize.TweetTokenizer(strip_handles=True, reduce_len=True)

    datasets = [
        'raw_data/train/train_airlines.csv',
        'raw_data/train/train_apple.csv',
        'raw_data/train/train_products.csv',
        'raw_data/test/test_airlines.csv',
        'raw_data/test/test_apple.csv',
        'raw_data/test/test_products.csv',
    ]

    data_tokens = []

    all_tokens = set()

    for dataset_path in datasets:
        with open(dataset_path, encoding='utf-8') as raw_file:
            with open(dataset_path.replace('raw', 'preprocessed'), 'w',
                      encoding='utf-8') as preprocessed_file:
                buffer = u''
                for line in raw_file.readlines():
                    if len(buffer) > 0:
                        line = buffer + line
                    try:
                        tweet, label = line.rsplit(',', 1)
                        label = label.rstrip('\n')
                        tweet = tweet.replace('\n', '').strip('"').rstrip('"')
                        tweet = tweet.lower()
                        tokens = set(tknzr.tokenize(tweet.encode('utf-8')))
                        for token in tokens:
                            all_tokens.add(token)
                        data_tokens.append((tokens, label))
                    except ValueError:
                        buffer = line

    all_tokens = list(nltk.FreqDist(token for token in all_tokens))[:2000]

    data = []

    for tokens, label in data_tokens:
        features = {}
        for token in all_tokens:
            features['contains({})'.format(token)] = (token in tokens)
        data.append((features, label))

    with open('data.pickle', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
