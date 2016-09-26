import json
import urllib
import urllib.request
import urllib.parse

def classify_tweet(tweet):

    data = json.dumps({'tweet': tweet}).encode()
    url = 'http://127.0.0.1:5000/classify'

    request = urllib.request.Request(url, data,
                                     {'Content-Type': 'application/json'})

    response = json.loads(urllib.request.urlopen(request).read().decode())

    return response['label']

def main():

    while True:
        tweet = input('tweet > ')
        print(classify_tweet(tweet))

if __name__ == '__main__':
    main()
