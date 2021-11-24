
from gensim.test.utils import common_texts
from gensim.models import Word2Vec

import os

import numpy as np


import nltk
nltk.download('stopwords')
nltk.download('punkt')



allfiles = []
for root, dirs, files in os.walk("/home/ubuntu/AutoLDA/Transcripts/", topdown=False):
    for name in files:
        if name.endswith(".txt") and "checkpoint" not in name:
            allfiles.append(os.path.join(root, name))


def getAllText(allfiles):
    res = []
    for file in allfiles:
        with open(file,"r") as f:
            listOfWord = f.read().split()
            res.append([word.lower() for word in listOfWord if word.isalpha()])
    return res
            
allText = getAllText(allfiles)


# print(len(allText))

def trainWord2Vec(allText):
    model = Word2Vec(sentences=allText, vector_size=100, window=5, min_count=1, workers=8)
    model.save("./Word2Vec/word2vec.model")
    print("Model saved.")
    # Store just the words + their trained embeddings.
    word_vectors = model.wv
    word_vectors.save("./Word2Vec/word2vec.wordvectors")
    print("Word vectors saved.")
    with open("./Word2Vec/word2vec.vocab", "w") as f:
        for word in word_vectors.key_to_index.keys():
            f.write(word + "\n")
    print("Vocab saved.")
    return model

def loadWord2Vec():
    model = Word2Vec.load("/home/ubuntu/AutoLDA/hyperband-master/Embeddings/Word2Vec/word2vec.model")
    word_vectors = model.wv
    return word_vectors


def getWord2Vec(word_vectors, word):
    try:
        return word_vectors[word]
    except KeyError:
        return np.zeros(100)


# trainWord2Vec(allText)

# wv = loadWord2Vec()
# print(wv.vocab.keys())

# print(getWord2Vec(loadWord2Vec(), "covid"))

word_vectors = loadWord2Vec()


def genEmbeddings_W2V(keyword):
    
    emb = getWord2Vec(word_vectors, keyword)
    return emb
