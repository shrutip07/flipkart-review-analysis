# 📁 Filename: app.py

import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w.isalpha() and w not in stop_words]
    return ' '.join(tokens)

st.set_page_config(page_title="Fake Review Detector", layout="centered")
st.title("🛍️ Flipkart Fake Review Detector (Live Training)")

uploaded_file = st.file_uploader("📁 Upload Flipkart Dataset (CSV with 'review' and 'rating' columns)", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    if "review" not in df.columns or "rating" not in df.columns:
        st.error("CSV must contain 'review' and 'rating' columns.")
    else:
        st.success("✅ Dataset loaded successfully.")

        # Preprocessing
        df["clean_review"] = df["review"].apply(clean_text)

        neg_words = ['disappointed', 'poor', 'bad', 'worst', 'terrible', 'awful', 'horrible']
        df["is_negative"] = df.apply(lambda row: row["rating"] < 4 or any(w in row["clean_review"] for w in neg_words), axis=1)

        X_text = df["clean_review"]
        y = df["is_negative"].astype(int)

        tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), min_df=3, max_df=0.85, sublinear_tf=True)
        X = tfidf.fit_transform(X_text)

        selector = SelectKBest(chi2, k=min(5000, X.shape[1]))
        X_selected = selector.fit_transform(X, y)

        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_selected, y)

        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42)

        model = LogisticRegression(class_weight="balanced", max_iter=300)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True)

        st.markdown("### 📊 Model Evaluation")
        st.write("**Accuracy:**", round(report["accuracy"], 4))
        st.write("**Precision:**", round(report["1"]["precision"], 4))
        st.write("**Recall:**", round(report["1"]["recall"], 4))
        st.write("**F1 Score:**", round(report["1"]["f1-score"], 4))

        st.markdown("---")
        st.markdown("### 📝 Try It Out")

        user_review = st.text_area("Enter a product review:")

        if st.button("Classify Review"):
            if user_review.strip() == "":
                st.warning("Please enter a review to classify.")
            else:
                cleaned = clean_text(user_review)
                vectorized = tfidf.transform([cleaned])
                selected = selector.transform(vectorized)
                prediction = model.predict(selected)[0]
                prob = model.predict_proba(selected)[0][1]

                if prediction == 1:
                    st.error(f"⚠️ **Likely Fake / Negative Review** (Confidence: {prob:.2f})")
                else:
                    st.success(f"✅ **Likely Genuine Review** (Confidence: {1 - prob:.2f})")
