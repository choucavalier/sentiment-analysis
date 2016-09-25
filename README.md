# Multinomial Naive Bayes Classifier for Tweet Sentiment Analysis

This code uses [`sklearn.naive_bayes.MultinomialNB`](http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
for classifying tweets in `[neutral, positive, negative]`.

## Installation

Before being being able to run the code, you'll have to insall a few dependencies.

##### Virtual Environment (with `pyenv`)

*If you already know about this, you should skip it.*

It is recommended to install everything in a virtual environment.

I like **pyenv** so here is how to do it with it:

1. Install [pyenv](https://github.com/yyuu/pyenv)
2. Install [pyenv-virtualenv](https://github.com/yyuu/pyenv-virtualenv)
3. `pyenv install 3.5.2` to install the right python version
4. `pyenv virtualenv 3.5.2 tweet-sentiment` to create a virtual environment for this project
5. `pyenv activate tweet-sentiment` to activate the virtualenv
6. `pip install --upgrade pip setuptools` just to make sure you have the last versions of `pip` and `setuptools`

##### Dependencies

- To install all the modules necessary, simply run
  ```
  pip install -r requirements.txt
  ```
  inside the virtualenv.

- `nltk.corpus.stopwords`: english stopwords

  you need to run the following in a python interpreter:
  ```
  >>> import nltk
  >>> nltk.download()
  ```
  This launches a GUI which allows you to download `corpus.stopwords` (just a few KBs).

## Usage

### Test the model

A vocabulary and a trained model are included in the repository, so you can simply run `predict.py` to test the model on manually typed sentences.

Example:

```python
$ python predict.py
tweet > i really love the idea of flying
positive
tweet > i spent 3 hours in the line before they told
        me there was no iphone left
negative
```

### Train a new model

To extract a vocabulary and features from the raw dataset, use `preprocess.py`. This script uses `pickle` to persist both in `vocabulary.pickle` and `data.pickle`. If the size of the vocabulary (set by `VOCABULARY_SIZE` in `preprocess.py`) is very large, `data.pickle` can reach a few GBs because of the increased size of the feature vector extracted for each data point (which has the size of the vocabulary).

Example:
```python
$ python preprocess.py 
... generating vocabulary
totally 295434 tokens of interest in the corpus
vocabulary size: 2000
10 most common words: ['sxsw', 'apple', 'mention', 'link', 'united', 
                       'rt', 'flight', 'co', 'usairways', 'americanair']
vocabulary saved in ./vocalubary.pickle
... extracting features
preprocessed data saved in ./data.pickle
```

`train.py` is used to train a model based on `vocabulary.pickle` with the extracted features saved by `preprocess.py` in `data.pickle`.

```python
$ python train.py 
... training model
18254 samples used for training
model MultinomialNB trained
... evaluating model
9127 samples used for testing
......................................................................
classification report
             precision    recall  f1-score   support

          0       0.72      0.60      0.66      3607
          1       0.57      0.49      0.53      1924
          2       0.70      0.87      0.78      3596

avg / total       0.68      0.69      0.68      9127

......................................................................
confusion matrix
[[2182  558  867]
 [ 521  937  466]
 [ 321  139 3136]]
......................................................................
model saved in ./model.pickle
```

## About

### Dataset

The dataset contains around **27K** tweets retrieved from
[Twitter](http://twitter.com) with user comments about airline companies and products, some containing positive and negative feedbacks.

### Choice of the model

### Preprocessing the data

### Training

### API