
import streamlit as st
import pandas as pd
import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

# Load models
model = joblib.load("model_logreg.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")
selector = joblib.load("feature_selector.pkl")

lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha()]
    return ' '.join(tokens)

st.title("🛒 Flipkart Review Sentiment Detector")
st.write("This app classifies reviews as Positive or Negative using a trained Logistic Regression model.")

review_input = st.text_area("Enter your Flipkart Review:")

if st.button("Classify"):
    if not review_input.strip():
        st.warning("Please enter a review.")
    else:
        cleaned = clean_text(review_input)
        vect = tfidf.transform([cleaned])
        selected = selector.transform(vect)
        pred = model.predict(selected)[0]
        prob = model.predict_proba(selected)[0][1]
        label = "Negative 😞" if pred == 1 else "Positive 😊"
        st.markdown(f"**Prediction:** {label}")
        st.markdown(f"**Confidence:** {prob*100:.2f}%")
