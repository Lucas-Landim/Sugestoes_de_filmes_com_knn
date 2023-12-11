# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:05:48 2023

@author: Lando
"""

##------------------------------------------------------------------------
## Autor: Prof. Roberto Angelo  (coding: utf-8 )
## Objetivo: Conceitos de Text Mining com Machine Learning
##------------------------------------------------------------------------

import numpy as np
import pandas as pd

################################################
# Carrega a base dados
################################################
dataset = pd.read_csv('base_final_filmes_v1.csv', sep=';')

# Ignorar avisos específicos
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

#------------------------------------------------------------------------------------------
# Informações complementares em:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#------------------------------------------------------------------------------------------
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token = RegexpTokenizer(r'\b[a-zA-Z]{2,}\b') # Expressão regular para remover simbolos, números e palavras com menos de 3 letras
cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', min_df=5, ngram_range=(1,2), tokenizer=token.tokenize) # max_features = 100
text_counts = cv.fit_transform(dataset['Text'])

# Converte a matriz esparsa em um DataFrame para melhor visualizar
text_counts = pd.DataFrame(text_counts.toarray(), columns=cv.get_feature_names_out())
print(text_counts)

##------------------------------------------------------------
## Separa os dados em treinamento e teste
##------------------------------------------------------------
y = dataset['Sentiment']   # Carrega alvo 
X = text_counts            # Carrega as colunas geradas
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

#---------------------------------------------------------------------------
## Ajusta modelo Naive Bayes com treinamento - Aprendizado supervisionado  
#---------------------------------------------------------------------------
from sklearn.naive_bayes import MultinomialNB
NaiveB = MultinomialNB()
NaiveB.fit(X_train, y_train)

#---------------------------------------------------------------------------
## Previsão usando os dados de teste
#---------------------------------------------------------------------------
# Naive Bayes
y_pred_test_NaiveB= NaiveB.predict(X_test)

#---------------------------------------------------------------------------
## Cálcula da Acurácia do Naive Bayes
#---------------------------------------------------------------------------
from sklearn import metrics
print()
print('----------------------------------------------------------')
print('Acurácia NaiveBayes:',metrics.accuracy_score(y_test, y_pred_test_NaiveB))
print('----------------------------------------------------------')