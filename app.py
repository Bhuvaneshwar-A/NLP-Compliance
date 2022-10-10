import pandas as pd
import streamlit as st
import nltk
import warnings
import string
import pickle
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer


nltk.download('punkt')
nltk.download("wordnet")
nltk.download("omw-1.4")
ps = PorterStemmer()
warnings.simplefilter("ignore")


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

st.title("Compliance Classifier")
st.write("Compliance Sample")
df = pd.read_excel("sn_compliance_control.xlsx")
st.write(df[['Description','Classification']].head())
input_sms = st.text_area("Enter the Queries")
st.write(" ")
if st.button("Predict"):
    # 1. preprocess
    transformed_sms = transform_text(input_sms)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_sms])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.write(" ")
        st.success("Preventive")
    else:
        st.write(" ")
        st.error("Detective")

    st.write(" ")
    st.write("Project by Bhuvaneshwar A")