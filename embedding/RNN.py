import embedding_with_glove
from plot import plot
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Bidirectional, LSTM, SimpleRNN, GRU


MAX_LEN = 100
DIMENSION = 300
MAX_WORDS = 10000
TEST_SIZE = 0.1
NUM_RNN = 300
RATE_DROP = 0.1
RNN_RATE_DROP = 0.5

def rnn_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, DIMENSION, weights = [embedding_matrix], input_length = MAX_LEN, trainable = False))
    model.add(Bidirectional(LSTM(NUM_RNN, dropout = RATE_DROP, recurrent_dropout = RNN_RATE_DROP)))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 10, batch_size = 256, validation_data = (X_test, Y_test))

    plot(history)


labels, texts = embedding_with_glove.load_text('/home/chloe/Downloads/imdb/train') 
texts_tokenized, word_index = embedding_with_glove.tokenize(texts)
X_train, X_test, Y_train, Y_test = embedding_with_glove.split(texts_tokenized, labels)
embedding_matrix = embedding_with_glove.embedding(word_index)

rnn_model()

