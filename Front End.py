#!/usr/bin/env python
# coding: utf-8

# In[26]:


import joblib
import streamlit as st

#loading model and count_vector
def load_svc():
    loaded_model = joblib.load("C:\\Users\\Vinay\\Sentiment Analysis\\best_svc_model.pkl")
    return loaded_model

def load_count_vector():
    loaded_model = joblib.load("C:\\Users\\Vinay\\Sentiment Analysis\\count_vectorizer.pkl")
    return loaded_model

st.title("Text Sentiment Analysis")

# Load the model
svc_model = load_svc()
vect = load_count_vector()

# User input
user_input = st.text_input("Enter text: ")

if user_input:
    # Make predictions using the loaded model
    res = vect.transform([user_input])

    prediction = svc_model.predict(res)
    st.write("Sentiment of Text is :", prediction)


# In[ ]:




