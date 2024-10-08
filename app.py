import pandas as pd
import streamlit as st 
import numpy as np 
import tensorflow as tf 
from tensorflow.keras.datasets import imdb 
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, SimpleRNN
from tensorflow.keras.models import load_model

# load the imdb dataset 
word_index=imdb.get_word_index()
reverse_word_index={value:key for key,value in word_index.items()}

# Load the model
model=load_model("SimpleRNN_imdb.h5") 


# Helper function
def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i-3, '?') for i in encoded_review])

# Function to process user inputs
def preprocess_text(text):
    words=text.lower().split()
    encoded_review=[word_index.get(word, 2)+3 for word in words]  # noqa: F841
    padded_review=sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# Streamlit app 

st.title("Sentiment Analysis for IMDB movie review")
st.write("Write down the movie review and the model will classify if the entered review is positive or negative")

# USer Input 
user_input=st.text_area("Movie Review")

review_history=[]
button=st.button("Classify the review")
if button:
    if user_input:
        with st.spinner("Processing your review........"):
            preprocess_input=preprocess_text(user_input)
            
            #Prediction 
            prediction=model.predict(preprocess_input)
            sentiment="Positive" if prediction[0][0] > 0.6 else "Negative"
            
        #Display the results 
        st.write(f"Sentiment of the movie : {sentiment}")
        st.write(f"Prediction Score : {prediction[0][0]}")
        review_history.append({"Review" : user_input,  "Sentiment" : sentiment, "Prediction Score" : prediction[0][0]})
    else:
        st.error("Please enter a movie review")
    
with st.expander("**:red[Spoiler Alert:] Check before you write your review**"):
    st.write("Our sentiment analysis model, doesn't care about your personal opinion, but rather meticulously analyzes the factual information you provide about the movie, such as plot points, character development, and cinematic craftsmanship, to accurately gauge the overall sentiment and emotional resonance of your review, all while ignoring your subjective biases and emotional outbursts"
             )
    
st.subheader("Sample Reviews")

col1, col2 = st.columns(2)

with col1:
    st.write("**Good Review**")
    st.text_area("Sample Good Review", """
    I loved this movie! The acting was superb, the plot was engaging, and the cinematography was breathtaking. I would highly recommend it to anyone who enjoys a good drama.
    """, height=200)

with col2:
    st.write("**Bad Review**")
    st.text_area("Sample Bad Review", """
    The movie is a disaster as the acting was very bad and the story is very predictable. The cinematography is not good.  I would not recommend it to anyone.

    """, height=200)
    
st.subheader(":red[Review History]")
if review_history:
    st.write(pd.DataFrame(review_history))
else:
    st.info("No reviews have been submitted")