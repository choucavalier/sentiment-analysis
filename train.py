import enum
import pickle
import random

import numpy as np
from sklearn import naive_bayes
from sklearn import metrics

label_to_int = {
    'neutral': 0,
    'positive': 1,
    'negative': 2,
}

int_to_label = {
    0: 'neutral',
    1: 'positive',
    2: 'negative',
}

def main():

    datasets = [
        'data/train/train_airlines.csv',
        'data/train/train_apple.csv',
        'data/train/train_products.csv',
        'data/test/test_airlines.csv',
        'data/test/test_apple.csv',
        'data/test/test_products.csv',
    ]

    with open('vocabulary.pickle', 'rb') as f:
        vocabulary = pickle.load(f)

    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    cleaned_data = []

    for features, label in data:
        if label in label_to_int:
            cleaned_data.append((features, label))

    random.shuffle(cleaned_data)

    n_samples = len(cleaned_data)
    n_features = len(vocabulary)

    x = np.ndarray(shape=(n_samples, n_features))
    y = np.ndarray(shape=(n_samples,), dtype=int)

    for i, datum in enumerate(cleaned_data):
        x[i] = datum[0]
        y[i] = label_to_int[datum[1]]

    n_training_samples = 2 * n_samples // 3

    x_train = x[:n_training_samples]
    y_train = y[:n_training_samples]

    x_test = x[n_training_samples:]
    y_test = y[n_training_samples:]

    model = naive_bayes.MultinomialNB()

    model.fit(x_train, y_train)

    expected = y_test
    predicted = model.predict(x_test)

    print(metrics.classification_report(expected, predicted))
    print(metrics.confusion_matrix(expected, predicted))

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

    print('model saved in model.pickle')

if __name__ == '__main__':
    main()
