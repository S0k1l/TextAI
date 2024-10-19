import pandas as pd
import re
import string
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
import pickle

with open('tfidf_vectorizer.pkl', 'rb') as file:
    vectorization = pickle.load(file)

model = tf.keras.models.load_model(r'text_AI_model.keras')

def wordopt(text):
    text = text.lower()
    text = re.sub('\[.*?\]', '', text)
    text = re.sub("\\W"," ",text) 
    text = re.sub('https?://\S+|www\.\S+', '', text)
    text = re.sub('<.*?>+', '', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('\n', '', text)
    text = re.sub('\w*\d\w*', '', text)    
    return text

classes = ["Fake", "True"]

while True:
    print('Enter text:')
    new_text = pd.Series([str(input())])
    new_text = new_text.apply(wordopt)

    xv_new_text = vectorization.transform(new_text)

    y_pred = model.predict(xv_new_text)

    y_pred_class = (y_pred > 0.5).astype("int32")

    print(f"Predicted Class: {classes[y_pred_class[0][0]]}")