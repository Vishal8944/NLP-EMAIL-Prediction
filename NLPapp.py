import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report
import streamlit as st

df = pd.read_csv("email.csv", encoding="utf-8")
nltk.download('stopwords')

df['Label'] = df['Label'].map({'Spam': 1, 'Non-Spam': 0})


def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    text = " ".join(word for word in text.split() if word not in stopwords.words('english'))  # Remove stopwords
    return text

df['Cleaned_Message'] = df['Message_body'].apply(clean_text)

X=df['Cleaned_Message']
Y=df['Label']

x_train, x_test, y_train, y_test = train_test_split(X,Y, test_size=0.25, random_state=12)
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

def predict_sms(text):
    cleaned_text = clean_text(text)
    prediction = model.predict([cleaned_text])[0]
    return "Spam" if prediction == 1 else "Non-Spam"

# Streamlit App
st.title("Email Spam Classifier")
st.write("Enter the email message to check if it's Spam or Non-Spam.")

# Input box for user to type message
user_input = st.text_area("Message", "")

# Predict button
if st.button("Predict"):
    if user_input:
        result = predict_sms(user_input)
        st.write(f"Prediction: **{result}**")
    else:
        st.write("Please enter a message to classify.")
