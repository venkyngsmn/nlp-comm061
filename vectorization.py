import numpy as np
import pandas as pd


from gensim.models import KeyedVectors
import gensim.downloader as api
from sklearn.feature_extraction.text import TfidfVectorizer

def Word2Vec(df):
    
    model = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    
    vectors = []
    for texts in df:
        word_vectors = [model[word.lower()] for word in texts if word.lower() in model]
        if word_vectors:
            word_vectors = np.mean(word_vectors, axis=0)
        else:
            word_vectors = np.zeros(model.vector_size)
        vectors.append(word_vectors)
    
    return np.array(vectors)

def GloVe(df):
    
    model = api.load('glove-twitter-200')
    
    vectors = []
    for texts in df:
        word_vectors = [model[word.lower()] for word in texts if word.lower() in model]
        if word_vectors:
            word_vectors = np.mean(word_vectors, axis=0)
        else:
            word_vectors = np.zeros(model.vector_size)
        vectors.append(word_vectors)
    
    return np.array(vectors)


