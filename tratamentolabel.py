import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import  classification_report, accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score


# leitura dos dados
dataset = pd.read_excel('dataset.xlsx')

colunas_irrelevantes = ['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']
dataset = dataset.drop(columns=colunas_irrelevantes)
dataset.replace(
    {
        'negative': 0, 
        'positive': 1,
    },
    inplace=True
)

resultado_exame = "SARS-Cov-2 exam result"
COL_RESULTADO_EXAME = dataset[resultado_exame]

THRESHOLD = 0.9

def transf_labelencoder(dataset):
    encoder = LabelEncoder()
    # instancia do ordinal encoder
    for coluna in dataset.columns:
        if coluna!= "SARS-Cov-2 exam result" and dataset[coluna].dtypes =='object':
            valores_antes = dataset[coluna].values
            tipo_antes_encoder =  dataset[coluna].dtypes
            dataset[coluna] = dataset[coluna].astype(str)
            dataset[coluna] = encoder.fit_transform(dataset[coluna].values)
            valores_depois = dataset[coluna].values
            # tipo_depois_encoder =  dataset[coluna].dtypes
            print(f'\n Coluna {coluna}: valores antes  {valores_antes} e valores depois {valores_depois} \n tipo antes {tipo_antes_encoder} e tipo depois {dataset[coluna].dtypes}')
    return dataset


def detect_and_handle_outliers(dataset):
    for coluna in dataset.columns:
        if dataset[coluna].dtypes != 'object':
            # Cálculo do IQR
            Q1 = dataset[coluna].quantile(0.25)
            Q3 = dataset[coluna].quantile(0.75)
            IQR = Q3 - Q1
            # Definição dos limites superior e inferior
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            # Identificação dos outliers
            outliers = dataset[(dataset[coluna] < lower_bound) | (dataset[coluna] > upper_bound)]
            
           
            # Tratamento dos outliers
            if not outliers.empty:
                # Substituir os outliers pelo valor mediano da coluna
                dataset[coluna] = np.where((dataset[coluna] < lower_bound) | (dataset[coluna] > upper_bound), dataset[coluna].median(), dataset[coluna])
    print(f'\n Outliers: {outliers}')
    return dataset

def preencher_mediana(dataset):
    return dataset.fillna(dataset.median())

def colunas_faltantes(dataset):
    percentual_dfaltantes = dataset.isna().sum() / len(dataset) * 100
    # Selecionando as colunas com mais de 90% de dados faltantes
    colunas_dfaltantes = percentual_dfaltantes[percentual_dfaltantes > 90].index
    cols_dfaltantes = dataset[colunas_dfaltantes]
    print(f'\n Colunas com mais de 90% de dados faltantes: { colunas_dfaltantes }')
    return cols_dfaltantes

def correlacao(correlacao_data, coluna_corr):
    return correlacao_data.corrwith(coluna_corr, numeric_only=True)

def correlacao_forte(result_correlacao, coluna_corr):
    colunas_correlacionadas = result_correlacao[result_correlacao.abs() > THRESHOLD].index
    print(f"Colunas com correlação acima de {THRESHOLD}: {colunas_correlacionadas}")    
    result_correlacao.plot(kind='bar', figsize=(10, 6))
    # Configuração dos rótulos dos eixos x e y e título do gráfico
    plt.xlabel('Variáveis', fontsize=12)
    plt.ylabel('Correlação', fontsize=12)
    plt.title(f'Correlação das colunas com {coluna_corr}', fontsize=14)
    # Adicionar legenda
    plt.legend(['Correlação'], loc='best', fontsize=10)
    # Exibir o gráfico
    plt.show()

def drop_dtfaltantes(dataset, colunas_drop):
    return dataset.drop(colunas_drop, axis=1)

def maiores_correlacaoes_abs(correlacao):
    print(f'colunas com maiores correlacao em valor absoluto{correlacao.abs().sort_values(ascending=False).head(10).index}') 
    index_correlacoes = correlacao.abs().sort_values(ascending=False).head(10).index
    return index_correlacoes

def maiores_correlacaoes(correlacao):
    print(f'colunas com maiores correlacao {correlacao.sort_values(ascending=False).head(10).index}') 
    index_correlacoes = correlacao.abs().sort_values(ascending=False).head(10).index
    return index_correlacoes

dataset = transf_labelencoder(dataset)
dataset = preencher_mediana(dataset)

datase_cols_faltantes = colunas_faltantes(dataset)
correlacao_faltantes = correlacao(datase_cols_faltantes, COL_RESULTADO_EXAME )
verificar_correlacao_dtfaltantes = correlacao_forte(correlacao_faltantes, COL_RESULTADO_EXAME)

dataset = drop_dtfaltantes(dataset, datase_cols_faltantes)

correlacao_all = correlacao(dataset, COL_RESULTADO_EXAME)

index_maiores_correlacoes_abs = maiores_correlacaoes_abs(correlacao_all)
index_maiores_correlacoes = maiores_correlacaoes(correlacao_all)

lista_imabs = index_maiores_correlacoes_abs.tolist()
lista_im = index_maiores_correlacoes.tolist()

print(f'lista {lista_imabs}')


X = dataset[lista_imabs]
y = dataset['SARS-Cov-2 exam result']   

print(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.23,  stratify=y, random_state=81)

print("Distribuição da coluna alvo no conjunto de treinamento:")
print(y_train.value_counts())

print("Distribuição da coluna alvo no conjunto de teste:")
print(y_test.value_counts())

print("Valores únicos da coluna alvo após a codificação:")
print(y.unique())
# knn = train_test_algortimos(KNeighborsClassifier(n_neighbors=5), nome, X_train, X_test, y_train, y_test )

knn = KNeighborsClassifier(n_neighbors=50)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Aplicar validação cruzada com 5 folds
scores = cross_val_score(dt, X, y, cv=5)

# Imprimir as acurácias obtidas em cada fold
print("Acurácias em cada fold:", scores)

# Imprimir a média das acurácias
print("Acurácia média:", scores.mean())

# Treinamento e teste dos dados com k-Means
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Imprimindo a acurácia de cada algoritmo
print("Acurácia da Árvore de Decisão:", dt_accuracy)
print("Acurácia do k-NN:", knn_accuracy)
print("Acurácia do naive bayes:", nb_accuracy)


prds = dt.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
print(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}', '\n\n',
      'Accuracy:', (accuracy_score(y_test, prds)), '\n\n',
      'Classification Report:\n', (classification_report(y_test, prds)))

prds = knn.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
print(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}', '\n\n',
      'Accuracy:', (accuracy_score(y_test, prds)), '\n\n',
      'Classification Report:\n', (classification_report(y_test, prds)))

prds = nb.predict(X_test)
tn, fp, fn, tp = confusion_matrix(y_test, prds).ravel()
print(f'tn {tn}, fp {fp}, fn {fn}, tp {tp}', '\n\n',
      'Accuracy:', (accuracy_score(y_test, prds)), '\n\n',
      'Classification Report:\n', (classification_report(y_test, prds)))




print(dataset)