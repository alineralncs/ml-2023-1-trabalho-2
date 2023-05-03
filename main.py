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

# leitura dos dados
dataset = pd.read_excel('dataset.xlsx')

colunas_irrelevantes = ['Patient ID', 'Patient addmited to regular ward (1=yes, 0=no)', 'Patient addmited to semi-intensive unit (1=yes, 0=no)', 'Patient addmited to intensive care unit (1=yes, 0=no)']
dataset = dataset.drop(columns=colunas_irrelevantes)

encoder = LabelEncoder()
# instancia do ordinal encoder
for coluna in dataset.columns:

    if dataset[coluna].dtypes =='object':
        valores_antes = dataset[coluna].values
        tipo_antes_encoder =  dataset[coluna].dtypes
        dataset[coluna] = dataset[coluna].astype(str)
        dataset[coluna] = encoder.fit_transform(dataset[coluna].values)
        valores_depois = dataset[coluna].values
        # tipo_depois_encoder =  dataset[coluna].dtypes
        print(f'\n Coluna {coluna}: valores antes  {valores_antes} e valores depois {valores_depois} \n tipo antes {tipo_antes_encoder} e tipo depois {dataset[coluna].dtypes}')

dataset = dataset.fillna(dataset.mean())


resultado_exame = "SARS-Cov-2 exam result"
selecionar_col_resultado = dataset[resultado_exame]
primeira_correlacao = dataset.corrwith(selecionar_col_resultado, numeric_only=True)

primeira_correlacao.plot(kind='bar', figsize=(10, 6))

# Configuração dos rótulos dos eixos x e y e título do gráfico
plt.xlabel('Variáveis', fontsize=12)
plt.ylabel('Correlação', fontsize=12)
plt.title('Correlação das colunas com mais de 90% dos dados faltantes com o resultado do exame', fontsize=14)

# Adicionar legenda
plt.legend(['Correlação'], loc='best', fontsize=10)

# Exibir o gráfico
plt.show()


percentual_dfaltantes = dataset.isna().sum() / len(dataset) * 100

# Selecionando as colunas com mais de 50% de dados faltantes
colunas_dfaltantes = percentual_dfaltantes[percentual_dfaltantes > 90].index

cols_dfaltantes = dataset[colunas_dfaltantes]
# Exibindo as colunas com mais de 90% de dados faltantes
print(f'\n Colunas com mais de 90% de dados faltantes: { colunas_dfaltantes }')

# verificar se essas colunas tem correlacao com o resultado do exame de covid
resultado_exame = "SARS-Cov-2 exam result"
selecionar_col_resultado = dataset[resultado_exame]

threshold = 0.9

alta_correlacao_dfaltantes = cols_dfaltantes.corrwith(selecionar_col_resultado, numeric_only=True)

colunas_correlacionadas = alta_correlacao_dfaltantes[alta_correlacao_dfaltantes.abs() > threshold].index
print(f"Colunas com correlação acima de {threshold}: {colunas_correlacionadas}")

alta_correlacao_dfaltantes.plot(kind='bar', figsize=(10, 6))

# Configuração dos rótulos dos eixos x e y e título do gráfico
plt.xlabel('Variáveis', fontsize=12)
plt.ylabel('Correlação', fontsize=12)
plt.title('Correlação das colunas com mais de 90% dos dados faltantes com o resultado do exame', fontsize=14)

# Adicionar legenda
plt.legend(['Correlação'], loc='best', fontsize=10)

# Exibir o gráfico
plt.show()

dataset = dataset.drop(colunas_dfaltantes, axis=1)



grupo1_correlacao = dataset.corrwith(selecionar_col_resultado, numeric_only=True)


print(f'colunas com maiores correlacao em valor absoluto{grupo1_correlacao.abs().sort_values(ascending=False).head(10).index}') 

index_correlacoes = grupo1_correlacao.abs().sort_values(ascending=False).head(10).index

# x = dataset[index_correlacoes[1:]]
# y = dataset[index_correlacoes[0]]
lista_im = index_correlacoes.tolist()

X = dataset[lista_im]
y = dataset['SARS-Cov-2 exam result']   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.23, random_state=101)

print(f'x train {X_train} e y train {y_train}' )
# Treinamento e teste dos dados com Árvore de Decisão
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_predictions = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_predictions)

# Treinamento e teste dos dados com k-NN
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_predictions)

# Treinamento e teste dos dados com k-Means
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_predictions)

# Imprimindo a acurácia de cada algoritmo
print("Acurácia da Árvore de Decisão:", dt_accuracy)
print("Acurácia do k-NN:", knn_accuracy)
print("Acurácia do naive bayes:", nb_accuracy)

algorithms = ['Decision Tree', 'k-NN', 'Naive Bayes']
accuracies = [dt_accuracy, knn_accuracy, nb_accuracy]

# Plotando o gráfico de barras com as acurácias
plt.bar(algorithms, accuracies)
plt.xlabel('Algoritmo')
plt.ylabel('Acurácia')
plt.title('Acurácia dos Algoritmos')
plt.ylim([0, 1])  # Definindo o limite do eixo y entre 0 e 1
plt.show()


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





