import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, GRU, Conv1D, GlobalMaxPooling1D, Dense, Dropout, Input, Concatenate
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from custom_losses import custom_loss

import sys
import os

# Añadir el directorio del proyecto al PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Directorio base del proyecto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Load preprocessed data
X_train_pad = np.load(os.path.join(BASE_DIR, 'data/X_train_pad.npy'))
X_val_pad = np.load(os.path.join(BASE_DIR, 'data/X_val_pad.npy'))
y_train = np.load(os.path.join(BASE_DIR, 'data/y_train.npy'))
y_val = np.load(os.path.join(BASE_DIR, 'data/y_val.npy'))

# Load saved tokenizer
tokenizer_path = os.path.join(BASE_DIR, 'data/tokenizer.json')
with open(tokenizer_path, 'r') as f:
    tokenizer_json = f.read()
    tokenizer = tokenizer_from_json(tokenizer_json)

# Common configuration
maxlen = 300
input_dim = len(tokenizer.word_index) + 1

# Load GloVe embeddings
embedding_index = {}
glove_path = os.path.join(BASE_DIR, 'data/glove.6B.100d.txt')
with open(glove_path, encoding="utf8") as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embedding_index[word] = coefs

# Prepare embedding matrix
embedding_dim = 100
embedding_matrix = np.zeros((input_dim, embedding_dim))
for word, i in tokenizer.word_index.items():
    embedding_vector = embedding_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Define models
def build_lstm_model(input_dim, maxlen, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        LSTM(64),
        Dense(1, activation='linear')
    ])
    return model

def build_gru_model(input_dim, maxlen, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        GRU(64),
        Dense(1, activation='linear')
    ])
    return model

def build_cnn_model(input_dim, maxlen, embedding_matrix):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False),
        Conv1D(filters=64, kernel_size=5, activation='relu'),
        GlobalMaxPooling1D(),
        Dense(1, activation='linear')
    ])
    return model

def build_hybrid_model(input_dim, maxlen, embedding_matrix):
    # Input layer
    inputs = Input(shape=(maxlen,))
    
    # Embedding layer
    embedding = Embedding(input_dim=input_dim, output_dim=embedding_dim, input_length=maxlen, weights=[embedding_matrix], trainable=False)(inputs)
    
    # LSTM layer
    lstm_out = LSTM(64, return_sequences=True)(embedding)
    lstm_out = GlobalMaxPooling1D()(lstm_out)
    
    # GRU layer
    gru_out = GRU(64, return_sequences=True)(embedding)
    gru_out = GlobalMaxPooling1D()(gru_out)
    
    # CNN layer
    conv_out = Conv1D(64, kernel_size=5, activation='relu')(embedding)
    conv_out = GlobalMaxPooling1D()(conv_out)
    
    # Concatenate LSTM, GRU, and CNN outputs
    concatenated = Concatenate()([lstm_out, gru_out, conv_out])
    
    # Fully connected layer
    dense_out = Dense(64, activation='relu')(concatenated)
    dense_out = Dropout(0.5)(dense_out)
    
    # Output layer
    outputs = Dense(1, activation='linear')(dense_out)
    
    # Create the model
    model = Model(inputs=inputs, outputs=outputs)
    
    return model

def build_random_model(input_dim, maxlen):
    model = Sequential([
        Embedding(input_dim=input_dim, output_dim=100, input_length=maxlen),
        LSTM(64),
        Dense(1, activation='linear')
    ])
    return model

# Train and evaluate models
def train_and_evaluate_model(model, X_train, y_train, X_val, y_val, model_name):
    model.compile(loss=custom_loss, optimizer='adam', metrics=['mae'])
    history = model.fit(X_train, y_train, batch_size=64, epochs=10, validation_data=(X_val, y_val))
    model_path = os.path.join(BASE_DIR, f'models/{model_name}.keras')
    model.save(model_path)
    return history

# Build LSTM model
lstm_model = build_lstm_model(input_dim, maxlen, embedding_matrix)
history_lstm = train_and_evaluate_model(lstm_model, X_train_pad, y_train, X_val_pad, y_val, 'lstm_model')

# Build GRU model
gru_model = build_gru_model(input_dim, maxlen, embedding_matrix)
history_gru = train_and_evaluate_model(gru_model, X_train_pad, y_train, X_val_pad, y_val, 'gru_model')

# Build CNN model
cnn_model = build_cnn_model(input_dim, maxlen, embedding_matrix)
history_cnn = train_and_evaluate_model(cnn_model, X_train_pad, y_train, X_val_pad, y_val, 'cnn_model')

# Build Hybrid model
hybrid_model = build_hybrid_model(input_dim, maxlen, embedding_matrix)
history_hybrid = train_and_evaluate_model(hybrid_model, X_train_pad, y_train, X_val_pad, y_val, 'hybrid_model')
