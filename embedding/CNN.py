import embedding_with_glove
from plot import plot
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, Conv1D, MaxPooling1D, GlobalAveragePooling1D, Model, Input, Dropout


MAX_LEN = 100
DIMENSION = 300
MAX_WORDS = 10000
TEST_SIZE = 0.1


def cnn_model():
    model = Sequential()
    model.add(Embedding(MAX_WORDS, DIMENSION, weights = [embedding_matrix], input_length = MAX_LEN, trainable = False))
    model.add(Conv1D(256, 3, activation = 'relu'))
    model.add(MaxPooling1D(pool_size=4))
    model.add(Conv1D(64, 3, activation = 'relu'))
    model.add(GlobalMaxPool1D())
    model.add(Dense(1, activation = 'sigmoid'))
    model.summary()
    
    model.compile(optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 10, batch_size = 128, validation_data = (X_train, Y_train))
    plot(history)

def cnn_model_functional():
    comment_input = Input(shape = (MAX_LEN, ), dtype = 'int32')
    embedded_sequences = Embedding(MAX_WORDS, DIMENSION, weights = [embedding_matrix], input_length = MAX_LEN, trainable = False)(comment_input)
    x = Conv1D(256, 3, activation = 'relu')(embedded_sequences)
    x = MaxPooling1D(5)(x)
    x = Dropout(0.2)(x)
    x = Conv1D(64, 3, activation = 'relu')(x)
    x = GlobalMaxPooling1D()(x)
    pred = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs = [comment_input], outputs = pred)
    model.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop', metrics = ['acc'])
    history = model.fit(X_train, Y_train, epochs = 10, batch_size = 128, validation_data = (X_test, Y_test))
    plot.plot(history)
    

labels, texts = embedding_with_glove.load_text('/home/chloe/Downloads/imdb/train') 
texts_tokenized, word_index = embedding_with_glove.tokenize(texts)
X_train, X_test, Y_train, Y_test = embedding_with_glove.split(texts_tokenized, labels)
embedding_matrix = embedding_with_glove.embedding(word_index)

cnn_model()
cnn_model_functional()
