# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 18:29:42 2023

@author: Pro. Roberto Santos
"""

import numpy as np
import pandas as pd

################################################
# Carrega a base dados
################################################
dataset = pd.read_csv('BASE.TXT', sep='\t')


################################################
# Separar conjuntos de TRN e TST
################################################
from sklearn.model_selection import train_test_split
train, test = train_test_split(dataset, test_size=0.25, random_state = 42)
train.insert(0,'CONJUNTO','TRN')    # Cria uma coluna com TRN
test.insert(0,'CONJUNTO','TST')    # Cria uma coluna com TST

dataset = pd.concat([train, test])


# (Opcional) Exportar a base de dados de TRN
dataset.to_csv('pre_processamento.csv',sep='\t')
print('Dados Exportados')


################################################
# Pré-processamento de Dados
################################################

# TIPO_CLIENTE
dataset['PRE_CLIENTE_EXPERIENTE'] = [1 if x== 'EXPERIENTE' else 0 for x in dataset['TIPO_CLIENTE']]
# dataset['PRE_CLIENTE_NEUTRO'] = [1 if x== 'NEUTRO' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_CLIENTE_NOVO'] = [1 if x== 'NOVO' else 0 for x in dataset['TIPO_CLIENTE']]
dataset['PRE_CLIENTE_VIP'] = [1 if x== 'VIP' else 0 for x in dataset['TIPO_CLIENTE']]

#CD_SEXO
dataset['PRE_SEXO_M'] = [1 if x== 'M' else 0 for x in dataset['CD_SEXO']]
dataset['PRE_SEXO_F_NULO'] = [1 if x=='F' or x==' ' else 0 for x in dataset['CD_SEXO']]

#IDADE
dataset['PRE_IDADE'] = [18 if np.isnan(x) or x < 18 else x for x in dataset['IDADE']] # tratar menores de 18 e nulos
dataset['PRE_IDADE'] = [73 if x > 73 else x for x in dataset['PRE_IDADE']]  # Tratar idades maiores que 73
dataset['PRE_IDADE'] = [(x-18)/(73-18) for x in dataset['PRE_IDADE']]

#BANCO
dataset['PRE_B1'] = [1 if x== 'B1' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B237'] = [1 if x== 'B237' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B341'] = [1 if x== 'B341' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B33'] = [1 if x== 'B33' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B104'] = [1 if x== 'B104' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B356'] = [1 if x== 'B356' else 0 for x in dataset['CD_BANCO']]
dataset['PRE_B399'] = [1 if x== 'B399' else 0 for x in dataset['CD_BANCO']]




# Alunos complementar com demais varriáveis

############################################################################
# SELECIONAR VARIÁVEIS
############################################################################

cols_in = ['PRE_CLIENTE_EXPERIENTE'
           ,'PRE_CLIENTE_NOVO'
           ,'PRE_CLIENTE_VIP'
           ,'PRE_SEXO_M'
           ,'PRE_SEXO_F_NULO'
           ,'PRE_IDADE'
           ,'PRE_B1'
           ,'PRE_B237'
           ,'PRE_B341'
           ,'PRE_B33'
           ,'PRE_B104'
           ,'PRE_B356'
           ,'PRE_B399'   
           ,'ALVO']

X_train = dataset.query("CONJUNTO == 'TRN'")
X_test = dataset.query("CONJUNTO == 'TST'")
X_train = X_train[cols_in]
X_test = X_test[cols_in]

y_train = X_train['ALVO']
y_test = X_test['ALVO']

del X_train['ALVO']
del X_test['ALVO']

#---------------------------------------------------------------------------
## Selecionando Atributos com RFE - Recursive Feature Elimination
#---------------------------------------------------------------------------
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

############################################################
## AJUSTAR/TREINAR OS MODELOS
############################################################
#Regressão Logística
from sklearn.linear_model import LogisticRegression
LogisticReg = LogisticRegression()
LogisticReg.fit(X_train, y_train)

#RNA
from sklearn.neural_network import MLPClassifier
RNA = MLPClassifier(hidden_layer_sizes=(5))
RNA.fit(X_train, y_train)

############################################################
## PREDIÇÃO
############################################################
#Regressão Logística
y_pred_train_RL = LogisticReg.predict(X_train)
y_pred_test_RL = LogisticReg.predict(X_test)
y_pred_train_RL_P = LogisticReg.predict_proba(X_train)
y_pred_test_RL_P = LogisticReg.predict_proba(X_test)

#RNA
y_pred_train_RNA = RNA.predict(X_train)
y_pred_test_RNA = RNA.predict(X_test)
y_pred_train_RNA_P = RNA.predict_proba(X_train)
y_pred_test_RNA_P = RNA.predict_proba(X_test)

###################################################################
## Montando um Data Frame (Matriz) com os resultados
###################################################################
# Conjunto de treinamento
df_train = pd.DataFrame(y_pred_train_RL, columns=['CLASSIF_RL'])
df_train['CLASSIF_RNA'] = y_pred_train_RNA
df_train['REGRESSION_RL'] = [x for x in y_pred_train_RL_P[:,1]] 
df_train['REGRESSION_RNA'] = [x for x in y_pred_train_RNA_P[:,1]]
df_train['TARGET'] = [x for x in y_train]
df_train['TRN_TST'] = 'TRAIN'
# Conjunto de test
df_test = pd.DataFrame(y_pred_test_RL, columns=['CLASSIF_RL'])
df_test['CLASSIF_RNA'] = y_pred_test_RNA
df_test['REGRESSION_RL'] = [x for x in y_pred_test_RL_P[:,1]]  
df_test['REGRESSION_RNA'] = [x for x in y_pred_test_RNA_P[:,1]]
df_test['TARGET'] = [x for x in y_test]
df_test['TRN_TST'] = 'TEST' 

# Juntando Conjunto de Teste e Treinamento
df_total = pd.concat([df_test, df_train])

## Exportando os dados para avaliação dos resultados
df_total.to_excel('saidas_modelos.xlsx')


###################################################################
## AVALIAÇÃO DE DESEMPENHO
###################################################################
from sklearn import metrics
Acc_RNA_Classificacao = metrics.accuracy_score(y_test, y_pred_test_RNA)
Acc_RL_Classificacao  = metrics.accuracy_score(y_test, y_pred_test_RL)
print()
print('----------------------------------------')
print('RNA Acurácia              :',np.round(Acc_RNA_Classificacao,4))
print('RNA Erro de Classificação :',np.round(1 - Acc_RNA_Classificacao,4))
print('RNA Erro Médio Quadrático :',np.round(np.mean((y_test - y_pred_test_RNA_P[:,1]) **2),4))
print()
print('RLog Acurácia             :',np.round(Acc_RL_Classificacao,4))
print('RLog Erro de Classificação:',np.round(1 - Acc_RL_Classificacao,4))
print('RLog Erro Médio Quadrático:',np.round(np.mean((y_test - y_pred_test_RL_P[:,1]) **2),4))
print('----------------------------------------')
print()
print()

###################################################################
## FÓRMULA REGRESSÃO LOGÍSTICA
###################################################################
print('Regressão Logística')
print('SCORE = ROUND(1/(1 + exp(-(       {:.10}'.format( str(LogisticReg.intercept_[0])))
for i in range(0, len(LogisticReg.coef_[0])):
    print('             + {:30}     *     {:.8}'.format(X_train.columns[i], str(LogisticReg.coef_[0,i])))
print('             ))) * 100,2) 	')