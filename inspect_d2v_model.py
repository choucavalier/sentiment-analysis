import numpy as np
from gensim.models import doc2vec

model = doc2vec.Doc2Vec.load('model.d2v')

print(len(model.docvecs))

print(model.docvecs.count)

# x = np.zeros((len(model.docvecs), model.syn0.shape[1]))

# for i in range(x.shape[0]):
    # x[i] = model.docvecs[i]

# print(x)

# while True:
    # word = input('word > ')
    # print(model.most_similar(word))
