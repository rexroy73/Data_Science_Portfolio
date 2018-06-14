# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:58:51 2018

@author: Prashita
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from wordcloud import WordCloud
from matplotlib import pyplot as plt

#avoiding unknown text error
df = pd.read_csv('spam.csv' , encoding='ISO-8859-1')

#drop columns not needed
df = df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis = 1)

#rename columns
df.columns = ['labels', 'data']

df['b_labels'] = df['labels'].map({'ham':0 , 'spam':1})
# defining output

Y = df['b_labels'].as_matrix()

#tfidf = TfidfVectorizer(decode_error = 'ignore')
#X = tfidf.fit_transform(df['data'])
count_vect = CountVectorizer(decode_error= 'ignore')
X = count_vect.fit_transform(df['data'])

Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size= 0.33)

model = MultinomialNB()
model.fit(Xtrain, Ytrain)

print('train score: ', model.score(Xtrain, Ytrain))
print('test score: ', model.score(Xtest, Ytest))

#visualize the data
def visualize(label):
    words = ''
    for msg in df[df['labels'] == label]['data']:
        msg = msg.lower()
        words += msg + ' '
    wordcloud = WordCloud(width = 600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()

visualize('spam')
visualize('ham')

#analysis of model showing its short comings
df['predictions'] = model.predict(X)

sneaked_spam = df[(df['predictions'] == 0) & (df['b_labels'] == 1)]['data']

#spam must items
for msg in sneak_spam:
    print(msg)

not_actual_spam = df[(df['predictions'] == 1) & (df['b_labels'] == 0)]['data']

#not spam items
for msg1 in not_actual_spam:
    print(msg1)

