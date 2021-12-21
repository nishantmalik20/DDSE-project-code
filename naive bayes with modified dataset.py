#!/usr/bin/env python
# coding: utf-8

# In[1]:


import re
import pandas as pd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import warnings
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

jsonList = []
SRC=[]
TGT=[]
count=0
print("Started Reading JSON file which contains multiple JSON document")
with open(r'C:\Users\Smrijay\OneDrive\Desktop\github-typo-corpus-master\NewJsonFileWithTypo.json', 'r', encoding='utf8') as jsonFile:
    for jsonObj in jsonFile:
        dictionary = json.loads(jsonObj)
        jsonList.append(dictionary)

print("Printing each JSON Decoded Object")
for obj in jsonList:
    key='is_typo'
    if key in obj['edits'][0]:
        SRC.append(obj['edits'][0]['src']['text'])
        TGT.append(obj['edits'][0]['which_type'])

#vectorization
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3000)
X = cv.fit_transform(SRC).toarray()
y=TGT



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state = 0)


print('Naive bayers')

from sklearn.naive_bayes import MultinomialNB
baseline_model = MultinomialNB().fit(X_train, y_train)

y_pred=baseline_model.predict(X_test)

from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))


# In[ ]:




