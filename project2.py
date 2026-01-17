import streamlit as st
import nltk
import spacy
import string
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem import PorterStemmer, LancasterStemmer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# DOWNLOAD NLTK DATA
nltk.download("punkt")
nltk.download("stopwords")

# LOAD SPACY MODEL
nlp = spacy.load("en_core_web_sm")

# STREAMLIT PAGE CONFIG
st.set_page_config(
    page_title="NLP Preprocessing App",
    layout="wide"
)

# APP TITLE
st.title("NLP Preprocessing App")
st.write(
    "Tokenization, Cleaning (with REGEX), Stemming, Lemmatization, "
    "Bag of Words, TF-IDF & Word Embeddings"
)

# USER INPUT
text = st.text_area(
    "Enter text for NLP processing",
    height=150,
    placeholder="Example: Aman is the HOD of HIT and loves NLP"
)

# SIDEBAR OPTIONS
option = st.sidebar.radio(
    "Select NLP Technique",
    [
        "Tokenization",
        "Text Cleaning (Regex)",
        "Stemming",
        "Lemmatization",
        "Bag of Words",
        "TF-IDF",
        "Word Embedding"
    ]
)

# PROCESS BUTTON
if st.button("Process Text"):
    if text.strip() == "":
        st.warning("Please enter some text.")

    #  TOKENIZATION 
    elif option == "Tokenization":
        st.subheader("Tokenization Output")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("### Sentence Tokenization")
            st.write(sent_tokenize(text))

        with col2:
            st.markdown("### Word Tokenization")
            st.write(word_tokenize(text))

        with col3:
            st.markdown("### Character Tokenization")
            st.write(list(text))

    #  TEXT CLEANING WITH REGEX 
    elif option == "Text Cleaning (Regex)":
        st.subheader("Text Cleaning Output (Using REGEX)")

        # Lowercase
        text_lower = text.lower()

        # REGEX Cleaning
        text_no_url = re.sub(r"http\S+|www\S+", "", text_lower)
        text_no_email = re.sub(r"\S+@\S+", "", text_no_url)
        text_no_numbers = re.sub(r"\d+", "", text_no_email)
        text_no_punct = re.sub(r"[^\w\s]", "", text_no_numbers)
        text_clean = re.sub(r"\s+", " ", text_no_punct).strip()

        # Stopword removal using spaCy
        doc = nlp(text_clean)
        final_words = [
            token.text for token in doc
            if not token.is_stop and token.text.strip() != ""
        ]

        st.markdown("### Original Text")
        st.write(text)

        st.markdown("### Cleaned Text")
        st.write(" ".join(final_words))

    #  STEMMING
    elif option == "Stemming":
        st.subheader("Stemming Output")

        words = word_tokenize(text)
        porter = PorterStemmer()
        lancaster = LancasterStemmer()

        df = pd.DataFrame({
            "Original Word": words,
            "Porter Stemmer": [porter.stem(w) for w in words],
            "Lancaster Stemmer": [lancaster.stem(w) for w in words]
        })

        st.dataframe(df, use_container_width=True)

    #  LEMMATIZATION 
    elif option == "Lemmatization":
        st.subheader("Lemmatization using spaCy")

        doc = nlp(text)
        data = [(token.text, token.pos_, token.lemma_) for token in doc]

        df = pd.DataFrame(data, columns=["Word", "POS", "Lemma"])
        st.dataframe(df, use_container_width=True)

    #  BAG OF WORDS 
    elif option == "Bag of Words":
        st.subheader("Bag of Words Representation")

        vectorizer = CountVectorizer()
        X = vectorizer.fit_transform([text])

        df = pd.DataFrame({
            "Word": vectorizer.get_feature_names_out(),
            "Frequency": X.toarray()[0]
        }).sort_values(by="Frequency", ascending=False)

        st.dataframe(df, use_container_width=True)

        st.markdown("### Word Frequency Pie Chart (Top 10)")
        df_top = df.head(10)

        fig, ax = plt.subplots()
        ax.pie(df_top["Frequency"], labels=df_top["Word"], autopct="%1.1f%%")
        ax.axis("equal")
        st.pyplot(fig)

    #  TF-IDF 
    elif option == "TF-IDF":
        st.subheader("TF-IDF Representation")

        tfidf = TfidfVectorizer()
        X = tfidf.fit_transform([text])

        df = pd.DataFrame({
            "Word": tfidf.get_feature_names_out(),
            "TF-IDF Score": X.toarray()[0]
        }).sort_values(by="TF-IDF Score", ascending=False)

        st.dataframe(df, use_container_width=True)

    #  WORD EMBEDDING 
    elif option == "Word Embedding":
        st.subheader("Word Embeddings using spaCy")

        doc = nlp(text)
        data = []

        for token in doc:
            if token.has_vector:
                data.append([
                    token.text,
                    token.vector_norm,
                    token.vector[:5]   # first 5 values only
                ])

        df = pd.DataFrame(
            data,
            columns=["Word", "Vector Magnitude", "Embedding (first 5 values)"]
        )

        st.dataframe(df, use_container_width=True)