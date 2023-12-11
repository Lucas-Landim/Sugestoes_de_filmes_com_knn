# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:17:42 2023

@author: Lando
"""

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer


################################################
# Carrega a base dados
################################################
dataset = pd.read_csv('base_final_filmes_v1.csv', sep=';')

# Ignorar avisos específicos
import warnings
warnings.simplefilter(action='ignore', category=UserWarning)

# Excluindo colunas que não serão utilizadas no projeto
colunas_para_excluir = ['type', 'director', 'date_added'] 
#Excluindo essas colunas pois não serão uteis para resolução do nosso projeto (possui muitos dados não categoricos)
dataset = dataset.drop(colunas_para_excluir, axis=1)

#Extrair o valor numérico da duração
dataset['duration'] = dataset['duration'].str.extract('(\d+)').astype(float)

# Substituindo todos os valores nulos por "N/A"
dataset.fillna("N/A", inplace=True)

################################################
# Pré-processamento de Dados
################################################

# COUNTRY
dataset['pre_country_null'] = [1 if x== 'N/A' else 0 for x in dataset['country']]
dataset['pre_country_us'] = [1 if 'United States' in x else 0 for x in dataset['country']]
dataset['pre_country_india'] = [1 if 'India' in x else 0 for x in dataset['country']]
dataset['pre_country_uk'] = [1 if 'United Kingdom' in x else 0 for x in dataset['country']]
dataset['pre_country_canada'] = [1 if 'Canada' in x else 0 for x in dataset['country']]
dataset['pre_country_spain'] = [1 if 'Spain' in x else 0 for x in dataset['country']]
dataset['pre_country_egypt'] = [1 if 'Egypt' in x else 0 for x in dataset['country']]
dataset['pre_country_nigeria'] = [1 if 'Nigeria' in x else 0 for x in dataset['country']]
dataset['pre_country_indonesia'] = [1 if 'Indonesia' in x else 0 for x in dataset['country']]
dataset['pre_country_france'] = [1 if 'France' in x else 0 for x in dataset['country']]
dataset['pre_country_turkey'] = [1 if 'Turkey' in x else 0 for x in dataset['country']]
dataset['pre_country_japan'] = [1 if 'Japan' in x else 0 for x in dataset['country']]
dataset['pre_country_philippines'] = [1 if 'Philippines' in x else 0 for x in dataset['country']]
dataset['pre_country_mexico'] = [1 if 'Mexico' in x else 0 for x in dataset['country']]
dataset['pre_country_brazil'] = [1 if 'Brazil' in x else 0 for x in dataset['country']]
dataset['pre_country_hk'] = [1 if 'Hong Kong' in x else 0 for x in dataset['country']]
dataset['pre_country_germany'] = [1 if 'Germany' in x else 0 for x in dataset['country']]

# RELEASE_YEAR
#como existem muitas decadas, vamos juntar todas antes dos anos 70
dataset['pre_year_under_1970'] = [1 if x > 1920 and x < 1980 else 0 for x in dataset['release_year']]
dataset['pre_year_under_1980'] = [1 if x > 1980 and x < 1990 else 0 for x in dataset['release_year']]
dataset['pre_year_under_1990'] = [1 if x > 1990 and x < 2000 else 0 for x in dataset['release_year']]
dataset['pre_year_under_2000'] = [1 if x > 2000 and x < 2010 else 0 for x in dataset['release_year']]
#como nosso dataset os anos vão até 2021 extendemos um pouco para evitar criar um dummie para um ano apenas
dataset['pre_year_under_2010'] = [1 if x > 2010 and x < 2021 else 0 for x in dataset['release_year']]

#RATING
dataset['pre_rating_null'] = [1 if x== 'UNRATED' else 0 for x in dataset['rating']]
dataset['pre_rating_all'] = [1 if x== 'ALL' else 0 for x in dataset['rating']]
dataset['pre_rating_under_11'] = [1 if x== '7+' or x== '11+' else 0 for x in dataset['rating']]
dataset['pre_rating_under_17'] = [1 if x== '12+' or x== '13+' or x=='14+' or x=='15+' or x=='16+' or x=='17+' else 0 for x in dataset['rating']]
dataset['pre_rating_above_18'] = [1 if x== '18+' else 0 for x in dataset['rating']]

#DURATION

# LISTED IN
dataset['pre_listedin_Action & Adventure'] = [1 if 'Action & Adventure' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Action'] = [1 if 'Action' in x or 'Action & Adventure' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Adventure'] = [1 if 'Adventure' in x or 'Action & Adventure' else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Animation'] = [1 if 'Animation' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Anime'] = [1 if 'Anime' in x or 'Anime Features' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Arthouse'] = [1 if 'Arthouse' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Arts'] = [1 if 'Arts' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Classic Movies'] = [1 if 'Classic Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Children & Family Movies'] = [1 if 'Children & Family Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Comedy'] = [1 if 'Comedy' in x or 'Comedies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Cult Movies'] = [1 if 'Cult Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Documentary'] = [1 if 'Documentary' or 'Documentaries' in x in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Drama'] = [1 if 'Drama' in x or 'Dramas' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Entertainment'] = [1 if 'Entertainment' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Faith and Spirituality'] = [1 if 'Faith and Spirituality' in x or 'Faith & Spirituality' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Fantasy'] = [1 if 'Fantasy' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Fitness'] = [1 if 'Fitness' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Historical'] = [1 if 'Historical' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Horror'] = [1 if 'Horror' in x or 'Horror Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Independent Movies'] = [1 if 'Independent Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_International'] = [1 if 'International' in x or 'International Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Kids'] = [1 if 'Kids' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_LGBTQ'] = [1 if 'LGBTQ Movies' in x or 'LGBTQ' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Military and War'] = [1 if 'Military and War' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Movies'] = [1 if 'Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Music & Musicals'] = [1 if 'Music & Musicals' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Music Videos and Concerts'] = [1 if 'Music Videos and Concerts' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Romance'] = [1 if 'Romance' in x or 'Romantic Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Science Fiction and Culture'] = [1 if 'Science Fiction and Culture' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Sci-Fi & Fantasy'] = [1 if 'Sci-Fi & Fantasy' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Special Interest'] = [1 if 'Special Interest' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Sports'] = [1 if 'Sports' in x or 'Sports Movies' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Stand-Up Comedy'] = [1 if 'Stand-Up Comedy' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Suspense'] = [1 if 'Suspense' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Talk Show and Variety'] = [1 if 'Talk Show and Variety' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Thrillers'] = [1 if 'Thrillers' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Unscripted'] = [1 if 'Unscripted' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Western'] = [1 if 'Western' in x else 0 for x in dataset['listed_in']]
dataset['pre_listedin_Young Adult Audience'] = [1 if 'Young Adult Audience' in x else 0 for x in dataset['listed_in']]

#SERVICE PLATAFORM
dataset['pre_plataform_prime'] = [1 if x== 'Prime Video' else 0 for x in dataset['Service Plataform']]
dataset['pre_plataform_netflix'] = [1 if x== 'Netflix' else 0 for x in dataset['Service Plataform']]


############################################################################
# SELECIONAR VARIÁVEIS
############################################################################

cols_in = ['pre_country_null',
'pre_country_us',
'pre_country_india',
'pre_country_uk',
'pre_country_canada',
'pre_country_spain',
'pre_country_egypt',
'pre_country_nigeria',
'pre_country_indonesia',
'pre_country_france',
'pre_country_turkey',
'pre_country_japan',
'pre_country_philippines',
'pre_country_mexico',
'pre_country_brazil',
'pre_country_hk',
'pre_country_germany',
'pre_year_under_1970',
'pre_year_under_1980',
'pre_year_under_1990',
'pre_year_under_2000',
'pre_year_under_2010',
'pre_rating_null',
'pre_rating_all',
'pre_rating_under_11',
'pre_rating_under_17',
'pre_rating_above_18',
'pre_listedin_Action & Adventure',
'pre_listedin_Action',
'pre_listedin_Adventure',
'pre_listedin_Animation',
'pre_listedin_Anime',
'pre_listedin_Arthouse',
'pre_listedin_Arts',
'pre_listedin_Classic Movies',
'pre_listedin_Children & Family Movies',
'pre_listedin_Comedy',
'pre_listedin_Cult Movies',
'pre_listedin_Documentary',
'pre_listedin_Drama',
'pre_listedin_Entertainment',
'pre_listedin_Faith and Spirituality',
'pre_listedin_Fantasy',
'pre_listedin_Fitness',
'pre_listedin_Historical',
'pre_listedin_Horror',
'pre_listedin_Independent Movies',
'pre_listedin_International',
'pre_listedin_Kids',
'pre_listedin_LGBTQ',
'pre_listedin_Military and War',
'pre_listedin_Movies',
'pre_listedin_Music & Musicals',
'pre_listedin_Music Videos and Concerts',
'pre_listedin_Romance',
'pre_listedin_Science Fiction and Culture',
'pre_listedin_Sci-Fi & Fantasy',
'pre_listedin_Special Interest',
'pre_listedin_Sports',
'pre_listedin_Stand-Up Comedy',
'pre_listedin_Suspense',
'pre_listedin_Talk Show and Variety',
'pre_listedin_Thrillers',
'pre_listedin_Unscripted',
'pre_listedin_Western',
'pre_listedin_Young Adult Audience',
'pre_plataform_prime',
'pre_plataform_netflix']



#------------------------------------------------------------------------------------------
# Informações complementares em:
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html
#------------------------------------------------------------------------------------------


# Considere que você já importou todas as bibliotecas necessárias

# Seu código para criar o tokenizador e o CountVectorizer
token = RegexpTokenizer(r'\b[a-zA-Z]{2,}\b')
cv = CountVectorizer(analyzer='word', lowercase=True, stop_words='english', min_df=5, ngram_range=(1,2), tokenizer=token.tokenize)

# Transformação das colunas 'title' e 'description'
title_counts = cv.fit_transform(dataset['title'])
description_counts = cv.transform(dataset['description'])  # Use transform em vez de fit_transform para a segunda transformação

# Converter as matrizes esparsas para DataFrames
title_counts_df = pd.DataFrame(title_counts.toarray(), columns=cv.get_feature_names_out())
description_counts_df = pd.DataFrame(description_counts.toarray(), columns=cv.get_feature_names_out())

# Concatenar os DataFrames
combined_data = pd.concat([cols_in, title_counts_df, description_counts_df])


#---------------------------------------------------------------------------
## Selecionando Atributos com RFE - Recursive Feature Elimination
#---------------------------------------------------------------------------

X_train = combined_data
X_test = combined_data
X_train = X_train[cols_in]
X_test = X_test[cols_in]

y_train = X_train
y_test = X_test



# feature extraction
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

model = LogisticRegression()
selected = RFE(model,step=1,n_features_to_select=10).fit(X_train, y_train)

print('--------------------------------------------------')
print('---  SELEÇÃO DE VARIÁVEIS  ---')
print('--------------------------------------------------')
print("Quantidade de Variáveis: %d" % selected.n_features_)
used_cols = []
for i in range(0, len(selected.support_)):
    if selected.support_[i]: 
        used_cols.append(X_train.columns[i]) 
        print('             -> {:30}     '.format(X_train.columns[i]))
print('--------------------------------------------------')

X_train = X_train[used_cols]     # Carrega colunas de entrada selecionadas por RFE
X_test = X_test[used_cols]       # Carrega colunas de entrada selecionadas por RFE


"""
##------------------------------------------------------------
## Separa os dados em treinamento e teste
##------------------------------------------------------------

#removendo todas colunas que podem haver dados em string
X = combined_data  
# Remove a coluna alvo
y = combined_data
# Carrega as colunas geradas
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
"""

