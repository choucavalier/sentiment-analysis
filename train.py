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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.utils import shuffle
from gensim.models import doc2vec

def main():

    # dict used to convert string labels to integers
    label_to_int = {
        'neutral': 0,
        'positive': 1,
        'negative': 2,
    }

    # load preprocessed data with shape (n_samples, DOC2VEC_SIZE)
    with open('data.pickle', 'rb') as f:
        x = pickle.load(f)

    # load labels
    with open('labels.pickle', 'rb') as f:
        y = pickle.load(f)

    n_samples, n_features = x.shape # for code clarity

    # shuffle x and y the same way
    x, y = shuffle(x, y, random_state=0)

    # splitting the dataset into train/test
    n_training_samples = 2 * n_samples // 3
    x_train = x[:n_training_samples]
    y_train = y[:n_training_samples]
    x_test = x[n_training_samples:]
    y_test = y[n_training_samples:]

    # hardcoded prior probabilities
    priors = np.array([0.5, 0.25, 0.25])

    model = LDA()

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
