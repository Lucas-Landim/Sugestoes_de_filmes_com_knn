# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 20:17:42 2023

@author: Lucas Landim
"""

import pandas as pd

################################################
# Carrega a base dados
################################################
dataset = pd.read_csv('base_final_filmes_v1.csv', sep=';')



# Excluindo colunas que não serão utilizadas no projeto
colunas_para_excluir = ['type', 'director', 'date_added']
dataset = dataset.drop(colunas_para_excluir, axis=1)


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

