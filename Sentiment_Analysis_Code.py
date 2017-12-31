# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:25:06 2017

@author: Kalyan
"""



def preprocess_Data(path):
    import pandas as pd
    import os
    
    #Navigating to the path where the datafile is kept
    basepath=path
    
    labels={'pos':1,'neg':1}
    df=pd.DataFrame()
    for s in ('test','train'):
        for l in ('pos','neg'):
            path=os.path.join(basepath,s,l)
            for file in os.listdir(path):
                with open (os.path.join(path,file),'r',encoding='utf-8') as infile:
                    txt=infile.read()
                df=df.append([[txt,labels[l]]],ignore_index=True)
    df.columns=['review','sentiment']
    
    #Randomizing the class labels and writing to a .csv file in the same directory
    import numpy as np
    np.random.seed(0)
    df=df.reindex(np.random.permutation(df.index))
    df.to_csv(('C:\\Users\\Kalyan\\Desktop\\NCSU\\1st Semester\\Courses\\IMDb_Sentiment_AnalysisProject\\aclImdb\\movie_data.csv'),index='False',encoding='utf-8')
    
    
    #Checking if the data has been read properly
    df=pd.read_csv(('C:\\Users\\Kalyan\\Desktop\\NCSU\\1st Semester\\Courses\\IMDb_Sentiment_AnalysisProject\\aclImdb\\movie_data.csv'),encoding='utf-8')
    return df
    
import re
def preprocessor(text):
    text=re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text



from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()

def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]

from nltk.corpus import stopwords
stop = stopwords.words('english')

 

filepath='C:\\Users\\Kalyan\\Desktop\\NCSU\\1st Semester\\Courses\\IMDb_Sentiment_AnalysisProject\\aclImdb'
df=preprocess_Data(filepath)
df['review']=df['review'].apply(preprocessor)
#print (df.loc[0,'review'][-50:])


#Dividing the data into Test and Training Data
X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values 

#Findind the optimal set of hyperparameters and learning the model
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV

tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)
    

#Fitting the training data
gs_lr_tfidf.fit(X_train, y_train)


#Printing the best parameeter set
print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)

#Fitting the Test Data with the optimal set of hyperparameeters
clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))





    
    