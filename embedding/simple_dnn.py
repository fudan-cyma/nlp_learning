import embedding_with_glove
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Bidirectional, LSTM, SimpleRNN, GRU


MAX_LEN = 100
DIMENSION = 300
MAX_WORDS = 10000
TEST_SIZE = 0.1


def rnn_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, DIMENSION, weights = [embedding_matrix], input_length = MAX_LEN, trainable = False))
    model.add(Flatten())
    model.add(Dense(32, activation = 'relu'))
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 10, batch_size = 256, validation_data = (X_test, Y_test))


labels, texts = embedding_with_glove.load_text('/home/chloe/Downloads/imdb/train') 
texts_tokenized, word_index = embedding_with_glove.tokenize(texts)
X_train, X_test, Y_train, Y_test = embedding_with_glove.split(texts_tokenized, labels)
embedding_matrix = embedding_with_glove.embedding(word_index)

model()

