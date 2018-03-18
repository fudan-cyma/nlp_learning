

import os 
import numpy as np
import pandas as pd 

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

MAX_LEN = 100
DIMENSION = 300
MAX_WORDS = 10000
TEST_SIZE = 0.1

glove_dir = '/home/chloe/Downloads/glove.840B.300d.txt'

def load_text():
    imdb_dir = '/home/chloe/Downloads/imdb'
    train_dir = os.path.join(imdb_dir, 'train')

    labels = []
    texts = []

    for label_type in ['neg', 'pos']:
        dir_name = os.path.join(train_dir, label_type)
        for fname in os.listdir(dir_name):
            if fname[-4:] == '.txt':
                f = open(os.path.join(dir_name, fname))
                texts.append(f.read())
                f.close()
                if label_type == 'neg':
                    labels.append(0)
                else:
                    labels.append(1)
    labels = np.asarray(labels)
    return labels, texts

def tokenize(texts):
    tokenizer = Tokenizer(num_words = MAX_WORDS)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    data = pad_sequences(sequences)
    return data

def split(texts_tokenized, labels):
    X_train, X_test, Y_train, Y_test = train_test_split(texts_tokenized, labels, test_size = TEST_SIZE)
    return X_train, X_test, Y_train, Y_test

def embedding():
    embedding_index = {}
    file = open(glove_dir)
    for line in file:
        values = line.split()
        #print(values)
        word = values[0]
        coefs = np.asarray(values[1:], dtype = 'float32')        
        embedding_index[word] = coefs
    file.close()
    
    embedding_matrix = np.zeros((MAX_WORDS, DIMENSION))
    for word, i in embedding_index:
        if i < MAX_WORDS:
            embedding_vector = embedding_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
    return embedding_matrix


    


labels, texts = load_text() 
texts_tokenized = tokenize(texts)
X_train, X_test, Y_train, Y_test = split(texts_tokenized, labels)
embedding_matrix = embedding()

