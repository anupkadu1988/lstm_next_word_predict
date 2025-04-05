## script for streamlit web app for LSTM next word predict application

import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

## load the LSTM model
model=load_model('next_word_lstm.h5')

## load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer=pickle.load(handle)

## function to predict the next word
def predict_next_word(model, tokenizer, text, max_sequence):
    token_list = tokenizer.texts_to_sequences([text])[0]
    if len(token_list) >= max_sequence:
        token_list = token_list[-(max_sequence-1):] ## Ensure the sequence legth matches max_sequence-1
    token_list=pad_sequences([token_list], maxlen=max_sequence-1, padding='pre')
    predicted=model.predict(token_list, verbose=0)
    predict_next_word = np.argmax(predicted, axis=1)
    for word, index in tokenizer.word_index.items():
        if index == predict_next_word:
            return word
    return None

## streamlit app
st.title("Next word prediction model using LSTM")
input_text=st.text_input("Enter the sequence of words")
if st.button("Click here to predict next word"):
    max_sequence=model.input_shape[1] + 1 ## retrieve the max sequence length
    next_word=predict_next_word(model, tokenizer, input_text, max_sequence)
    st.write(f"Net word: {next_word}")