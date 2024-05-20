import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import pickle

# --- Page 1: Authentication ---
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'page' not in st.session_state:
    st.session_state.page = 'authentication'

if st.session_state.page == 'authentication':
    st.set_page_config(page_title="Authentication")
    st.title("Authentication")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "12345":
            st.success("Login successful!")
            st.session_state.authenticated = True
            st.session_state.page = 'prediction'
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")

# --- Page 2: ML Model Prediction ---
if st.session_state.authenticated and st.session_state.page == 'prediction':
    st.set_page_config(page_title="Spam Detection")
    st.title("Spam Detection")

    # Load the saved model and vectorizer
    filename = 'spam_detection_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))

    filename = 'tfidf_vectorizer.sav'
    loaded_vectorizer = pickle.load(open(filename, 'rb'))

    # Input text box
    input_text = st.text_area("Enter your SMS message:", height=100)

    if st.button("Predict"):
        if input_text:
            # Preprocess the input text
            input_text = input_text.lower()  # Convert to lowercase
            input_text = input_text.replace("[^a-zA-Z0-9 ]", "")  # Remove punctuation

            # Transform the input text using the loaded vectorizer
            input_vector = loaded_vectorizer.transform([input_text])

            # Make prediction using the loaded model
            prediction = loaded_model.predict(input_vector)[0]

            # Display the prediction
            if prediction == "ham":
                st.success("This message is not spam.")
            else:
                st.warning("This message is likely spam.")
        else:
            st.warning("Please enter an SMS message.")