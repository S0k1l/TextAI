import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import re
import string

df_fake = pd.read_csv("TextAI/Fake.csv")
df_true = pd.read_csv("TextAI/True.csv")

df_fake["class"] = 0
df_true["class"] = 1

df_fake_manual_testing = df_fake.tail(10)
for i in range(23480,23470,-1):
    df_fake.drop([i], axis = 0, inplace = True)
    
    
df_true_manual_testing = df_true.tail(10)
for i in range(21416,21406,-1):
    df_true.drop([i], axis = 0, inplace = True)

df_fake_manual_testing["class"] = 0
df_true_manual_testing["class"] = 1

df_manual_testing = pd.concat([df_fake_manual_testing,df_true_manual_testing], axis = 0)
df_manual_testing.to_csv("TextAI/manual_testing.csv")

df_merge = pd.concat([df_fake, df_true], axis =0 )

df = df_merge.drop(["title", "subject","date"], axis = 1)

df = df.sample(frac = 1)

df.reset_index(inplace = True)
df.drop(["index"], axis = 1, inplace = True)

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

df["text"] = df["text"].apply(wordopt)

x = df["text"]
y = df["class"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

import pickle

with open('tfidf_vectorizer.pkl', 'wb') as file:
    pickle.dump(vectorization, file)

import tensorflow as tf

input_shape = (xv_train.shape[1],) 
input_layer = tf.keras.layers.Input(shape=input_shape)

base_layer = tf.keras.layers.Dense(128, activation='relu')(input_layer)
base_layer = tf.keras.layers.Dense(64, activation='relu')(base_layer)

classifier_branch = tf.keras.layers.Dense(32, activation='relu')(base_layer)
classifier_branch = tf.keras.layers.Dense(1, activation='sigmoid', name='cl_head')(classifier_branch)

model = tf.keras.Model(inputs=input_layer, outputs=classifier_branch)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(xv_train, y_train, epochs=5, batch_size=64, validation_data=(xv_test, y_test))

loss, accuracy = model.evaluate(xv_test, y_test)
print(f"Accuracy: {accuracy}")

model.save('text_AI_model.keras')

import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

plt.tight_layout()
plt.show()