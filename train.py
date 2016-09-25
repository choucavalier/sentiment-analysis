'''Train the model on preprocessed data

This module:

1.  generates a model trained on preprocessed data

2.  evaluates the model using 10-fold cross validation
'''
import enum
import pickle
import random

import numpy as np
from sklearn import cross_validation
from sklearn import naive_bayes
from sklearn import metrics


def main():

    # dict used to convert string labels to integers
    label_to_int = {
        'neutral': 0,
        'positive': 1,
        'negative': 2,
    }

    # load the preprocessed data (see preprocess.py)
    with open('data.pickle', 'rb') as f:
        data = pickle.load(f)

    cleaned_data = []

    # shuffling the data
    random.shuffle(data)

    n_samples = len(data)
    n_features = data[0][0].shape[0]

    # create ndarrays from preprocessed data
    x = np.ndarray(shape=(n_samples, n_features))
    y = np.ndarray(shape=(n_samples,), dtype=int)
    for i, datum in enumerate(data):
        x[i] = datum[0]
        y[i] = label_to_int[datum[1]]

    # splitting the dataset into train/test
    n_training_samples = 2 * n_samples // 3
    x_train = x[:n_training_samples]
    y_train = y[:n_training_samples]
    x_test = x[n_training_samples:]
    y_test = y[n_training_samples:]

    # hardcoded prior probabilities
    priors = np.array([0.5, 0.25, 0.25])

    model = naive_bayes.MultinomialNB(class_prior=priors)

    print(x_train.shape[0], 'samples used for training/validation')

    print('... fitting model')

    model.fit(x_train, y_train)

    print('done')

    print('... cross validition')

    scores = cross_validation.cross_val_score(model, x_train, y_train, cv=10)
    print('accuracy: {} (+/- {})'.format(scores.mean(), scores.std() * 2))

    print('... testing model')
    print(x_test.shape[0], 'samples used for testing')

    expected = y_test
    predicted = cross_validation.cross_val_predict(model, x_test, y_test, cv=10)

    print('.' * 70)

    print('classification report')

    print(metrics.classification_report(expected, predicted))

    print('.' * 70)

    print('confusion matrix')
    print(metrics.confusion_matrix(expected, predicted))

    print('.' * 70)

    with open('model.pickle', 'wb') as f:
        pickle.dump(model, f)

    print('model saved in', './model.pickle')

if __name__ == '__main__':
    main()
