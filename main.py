import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

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

# dataset = dataset.fillna(dataset.mean())

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


print(dataset.columns)

colunas = dataset.columns

grupo1 = dataset.iloc[:, :len(colunas)//2]
grupo2 = dataset.iloc[:, len(colunas)//2:]


grupo1_correlacao = dataset.corrwith(selecionar_col_resultado, numeric_only=True)


print(f'colunas com maiores correlacao em valor absoluto{grupo1_correlacao.abs().sort_values(ascending=False).head(20).index}') 


