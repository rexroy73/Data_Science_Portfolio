# -*- coding: utf-8 -*-
"""
Created on Wed Jun  6 17:58:51 2018

@author: Prashita
"""

from sklearn.naive_bayes import MultinomialNB

data= pd.read_csv('spambase.data').as_matrix()

np.random.shuffle(data)

#Dataset
X= data[:, :48]
Y= data[:, -1]

#Division of dataset
Xtrain = X[:-100,]
Ytrain = Y[:-100,]
Xtest = X[-100:,]
Ytest = Y[-100:,]

#fitting the model 1 --- Naive Bayes Classifier
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print ("Classification rate for NB:", model.score(Xtest, Ytest))


from sklearn.ensemble import AdaBoostClassifier

#fitting the model 1 --- AdaBoost Classifier
model1 = AdaBoostClassifier()
model.fit(Xtrain, Ytrain)
print ("Classification rate for Adaboost:", model.score(Xtest, Ytest))
