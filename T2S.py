import streamlit as st
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# ---------- Sample dataset ----------
data = [
    ("cat", "k ae t"),
    ("dog", "d ao g"),
    ("hello", "hh eh l ow"),
    ("world", "w er l d"),
    ("apple", "ae p l"),
]

df = pd.DataFrame(data, columns=["word", "phonemes"])

# ---------- Encoding ----------
chars = list(set("".join(df["word"].values)))
phonemes = list(set(" ".join(df["phonemes"].values).split()))

char_encoder = OneHotEncoder(sparse_output=False)
phoneme_encoder = OneHotEncoder(sparse_output=False)

char_encoder.fit(np.array(chars).reshape(-1, 1))
phoneme_encoder.fit(np.array(phonemes).reshape(-1, 1))

# ---------- Training Data ----------
X = []
y = []

for word, phoneme_seq in zip(df["word"], df["phonemes"]):
    for char, phoneme in zip(word, phoneme_seq.split()):
        X.append(char_encoder.transform([[char]])[0])
        y.append(phoneme_encoder.transform([[phoneme]])[0])

X = np.array(X)
y = np.array(y)

# ---------- Model ----------
model = Sequential()
model.add(Dense(64, input_shape=(X.shape[1],), activation='relu'))
model.add(Dense(y.shape[1], activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X, y, epochs=300, verbose=0)

# ---------- Prediction Function ----------
def predict_phonemes(word):
    output = []
    for char in word:
        try:
            encoded = char_encoder.transform([[char]])[0].reshape(1, -1)
            prediction = model.predict(encoded, verbose=0)
            phoneme = phoneme_encoder.inverse_transform(prediction)[0][0]
            output.append(phoneme)
        except:
            output.append('?')
    return " ".join(output)

# ---------- Speak Function ----------
def speak_phonemes(phoneme_string):
    os.system(f'espeak -v en "{phoneme_string}"')

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Text to Speech (NETTalk)", layout="centered")
st.title("🗣️ English Text to Speech (Inspired by NETTalk)")
st.markdown("Enter a word and hear it spoken using phoneme prediction!")

user_input = st.text_input("Enter a word:", max_chars=10)

if user_input:
    predicted = predict_phonemes(user_input.lower())
    st.write("🔤 **Predicted Phonemes:**", predicted)
    
    if st.button("🔊 Speak"):
        speak_phonemes(predicted)
