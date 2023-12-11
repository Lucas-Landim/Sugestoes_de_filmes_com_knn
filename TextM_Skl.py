##------------------------------------------------------------------------
## Autor: Prof. Roberto Angelo  (coding: utf-8 )
## Objetivo: Conceitos de Text Mining com Machine Learning
##------------------------------------------------------------------------

# Bibliotecas padrão  e carga de dados
import pandas as pd
dataset = pd.read_csv('Amazon_Reviews_1000.txt', sep=';') 

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
cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', min_df=10, ngram_range=(1,1), tokenizer=token.tokenize) # max_features = 100
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


