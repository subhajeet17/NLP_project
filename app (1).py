import streamlit as st
import nltk
import spacy
import pandas as pd
import matplotlib.pyplot as plt
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Download NLTK resources
nltk.download("punkt")
nltk.download("stopwords")

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Streamlit config
st.set_page_config(page_title="NLP Preprocessing App", layout="wide")

st.title("NLP Preprocessing Application")
st.write("Tokenization | Cleaning | Stemming | Lemmatization | BoW | TF-IDF | Embeddings")

text = st.text_area("Enter text", height=150)

option = st.sidebar.radio(
    "Select NLP Task",
    [
        "Tokenization",
        "Text Cleaning",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embedding"
    ]
)

if st.button("Process"):
    if option == "Tokenization":
        st.write("### Sentence Tokens", sent_tokenize(text))
        st.write("### Word Tokens", word_tokenize(text))

    elif option == "Text Cleaning":
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|\S+@\S+|\d+|[^\w\s]", "", text)
        doc = nlp(text)
        cleaned = [t.text for t in doc if not t.is_stop]
        st.write(" ".join(cleaned))

    elif option == "Stemming":
        porter = PorterStemmer()
        lancaster = LancasterStemmer()
        words = word_tokenize(text)
        df = pd.DataFrame({
            "Word": words,
            "Porter": [porter.stem(w) for w in words],
            "Lancaster": [lancaster.stem(w) for w in words]
        })
        st.dataframe(df)

    elif option == "Lemmatization":
        doc = nlp(text)
        df = pd.DataFrame([(t.text, t.lemma_) for t in doc],
                          columns=["Word", "Lemma"])
        st.dataframe(df)

    elif option == "Bag of Words":
        cv = CountVectorizer()
        X = cv.fit_transform([text])
        df = pd.DataFrame({
            "Word": cv.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        })
        st.dataframe(df)

    elif option == "TF-IDF":
        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([text])
        df = pd.DataFrame({
            "Word": tfidf.get_feature_names_out(),
            "Score": X.toarray()[0]
        })
        st.dataframe(df)

    elif option == "Word Embedding":
        doc = nlp(text)
        data = [(t.text, t.vector_norm) for t in doc if t.has_vector]
        st.dataframe(pd.DataFrame(data, columns=["Word", "Vector Magnitude"]))
