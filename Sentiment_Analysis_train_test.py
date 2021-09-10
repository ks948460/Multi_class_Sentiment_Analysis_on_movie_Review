# -*- coding: utf-8 -*-
"""
Created on Fri Sep 10 18:06:50 2021

@author: Krishna Sharma
"""

import pandas as pd
import numpy as np
import re
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Bidirectional,LSTM
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


#Reading csv file
data=pd.read_csv("Sentiment_Data.tsv",delimiter="\t")

#Droping unnecessary  
data = data.drop(['PhraseId','SentenceId'],axis = 1)

#Cleaning data and removing the redundency
def text_cleaning(text):
    if text:
        text = ' '.join(text.split('.'))
        text = re.sub('\/',' ',text)
        text = re.sub(r'\\',' ',text)
        text = re.sub(r'((http)\S+)','',text)
        text = re.sub(r'\s+', ' ', re.sub('[^A-Za-z]', ' ', text.strip().lower())).strip()
        text = re.sub(r'\W+', ' ', text.strip().lower()).strip()
        text = [word for word in text.split() ]
        return text
    return []

#applying the text_cleaning function to the phrase column
data['Phrase'] = data['Phrase'].apply(lambda x: ' '.join(text_cleaning(x)))

#Extracting the X and y
X_df=data['Phrase'] 
y_df=data['Sentiment']
X_values=X_df.values
y_values=y_df.values

#determining the maximum length for padding
max_len=max([len(x) for x in X_values])

#Encoding the word to numbers
vocab_size=4000
encoded_X = [one_hot(d, vocab_size) for d in X_values]

#Padding sequences to deal with inequal length of data
X=pad_sequences(encoded_X,maxlen=max_len,padding='post')

#One-Hot encodingon label_data
encoder = OneHotEncoder(sparse=False)
y = encoder.fit_transform(y_df.reshape(-1, 1))

#splitting the data as train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


#Defining the model with word embedding by keras 
embeded_vector_size=50

model = Sequential()
model.add(Embedding(vocab_size, embeded_vector_size, input_length=max_len,name="embedding"))
model.add(Bidirectional(LSTM(256, return_sequences=True),name="Lstm1"))
model.add(Bidirectional(LSTM(128, return_sequences=False),name="Lstm2"))
model.add(Dense(5, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
model.fit(X_train, y_train, epochs=5,batch_size=64,validation_split=0.2,verbose=1)
print(model.summary())


#predicting the output
y_pred=model.predict(X_test) 

#As the label is one-hot encoded, We have to convert it as single value
y_test=np.argmax(y_test,axis=1)
y_pred=np.argmax(y_pred,axis=1)

#printing the classification report
report=classification_report(y_test,y_pred)
print(report)

#printing the confusion marix
confusion_report=confusion_matrix(y_test,y_pred)
print(confusion_report)