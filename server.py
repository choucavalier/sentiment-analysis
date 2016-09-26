import flask
from flask import Flask, request
from flask import render_template

from classify import Classifier

app = Flask(__name__)

classifier = Classifier()

# keep track of all classified tweets to display them
history = []

history.append(('prisca', 'positive'))

@app.route('/', methods=['GET'])
def tweets_history():

    return render_template('history.html', history=list(reversed(history))[:10])

@app.route('/classify', methods=['POST'])
def classify_tweet():

    data = request.get_json()

    tweet = data['tweet']
    label = classifier.classify(tweet)

    history.append((tweet, label))

    return flask.jsonify(**{'label': label})
