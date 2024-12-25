import streamlit as st
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()

def transform_text(msg):
    msg = msg.lower()
    msg = nltk.word_tokenize(msg)

    y=[]
    for i in msg:
        if i.isalnum():
            y.append(i)
    msg=y[:]   
    y.clear()

    for i in msg:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
    msg=y[:]
    y.clear()

    for i in msg:
        y.append(ps.stem(i))
        
    return " ".join(y)


tfidf = pickle.load(open('vectorizer.pkl','rb'))
model = pickle.load(open('model.pkl','rb'))

st.title("Email/SMS Spam Detector")

input_msg = st.text_area("Enter the Message")

if st.button('Predict'):
    transformed_msg = transform_text(input_msg)
    
    vector_input = tfidf.transform([transformed_msg])
    
    result = model.predict(vector_input)[0]
    
    if result==1:
        st.header("Spam")
    else:
        st.header("Not Spam")