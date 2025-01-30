# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.models import Sequential
import h5py
import json
import keras
from keras.saving import deserialize_keras_object
from keras.layers import Embedding, SimpleRNN, Dense

# Register required custom objects for Keras deserialization
custom_objects = {
    "Sequential": Sequential,
    "Embedding": Embedding,
    "SimpleRNN": SimpleRNN,
    "Dense": Dense
}

# Define the file path
model_path = 'simple_rnn_imdb.h5'

# Open the .h5 file
with h5py.File(model_path, 'r') as f:
    config = json.loads(f.attrs['model_config'])

# Edit the configuration to remove 'time_major'
for layer in config['config']['layers']:
    if layer['class_name'] == 'SimpleRNN':
        layer['config'].pop('time_major', None)

# Reconstruct the model
model = deserialize_keras_object(config, custom_objects=custom_objects)
model.load_weights('simple_rnn_imdb.h5')


# Load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value + 3: key for key, value in word_index.items()}


# Step 2: Helper Functions
# Function to decode reviews
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i, '?') for i in encoded_review])

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review


import streamlit as st
## streamlit app
# Streamlit app
st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as positive or negative.')

# User input
user_input = st.text_area('Movie Review')

if st.button('Classify'):

    preprocessed_input=preprocess_text(user_input)

    ## MAke prediction
    prediction=model.predict(preprocessed_input)
    sentiment='Positive' if prediction[0][0] > 0.5 else 'Negative'

    # Display the result
    st.write(f'Sentiment: {sentiment}')
    st.write(f'Prediction Score: {prediction[0][0]}')
else:
    st.write('Please enter a movie review.')

