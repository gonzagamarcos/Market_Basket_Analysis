# Autor: Marcos Vinícius Gonzaga
# Projeto: Market Basket Analysis - Online Retail Data Set
# Dataset: Online Retail Data Set
# Link: https://archive.ics.uci.edu/ml/datasets/online+retail

# Objetivo:  Parte 1:
# Analisar os padrões de co-ocorrência entre os produtos comprados pelos clientes. Identificando quais produtos
# são frequentemente comprados juntos em uma mesma transação.

# Parte 2:
# Identificar quais produtos são mais frequentemente comprados em conjunto por determinados segmentos de clientes.

# O que esperar: Essa análise pode ser útil para entender melhor o comportamento dos clientes, criar promoções
# personalizadas, otimizar o layout da loja online, entre outros objetivos.


# Dicionário de Dados:

# InvoiceNo: Invoice number. Nominal, a 6-digit integral number uniquely assigned to each transaction. If this code starts with letter 'c', it indicates a cancellation.
# StockCode: Product (item) code. Nominal, a 5-digit integral number uniquely assigned to each distinct product.
# Description: Product (item) name. Nominal.
# Quantity: The quantities of each product (item) per transaction. Numeric.
# InvoiceDate: Invice Date and time. Numeric, the day and time when each transaction was generated.
# UnitPrice: Unit price. Numeric, Product price per unit in sterling.
# CustomerID: Customer number. Nominal, a 5-digit integral number uniquely assigned to each customer.
# Country: Country name. Nominal, the name of the country where each customer resides.


# Importando Bibliotecas
import pandas as pd

# carregamento dos dados
import pandas as pd # Manipulação de Dados

# Abaixo: funções são usadas para executar a análise de regras de associação com o algoritmo Apriori.
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Carregando o Dataset em uma variável
# leitura do arquivo de dados
#df = pd.read_excel('Online Retail.xlsx')


# Para otimizar o tempo de execução vou salvar o dataset carregando na memória, assim ele não será carregado toda vez que
# o programa for executado, tornando um pouco mais rápida que o metódo tradicional.


# Salvando o DataFrame em um arquivo .pkl
#df.to_pickle('online_retail.pkl')


# Carregando o DataFrame do arquivo .pkl
df = pd.read_pickle('online_retail.pkl')

# Visualizando informações sobre o conjunto de dados
print("Visualizando informações sobre o conjunto de dados")
print(df.shape) # 541909 x 8
print()


print(df.dtypes)
print()


print(df.head(5))
print()


# Retornando os valores de algumas variáveis
print("Retornando os valores únicos de algumas variáveis")
print("Country:")
print(df["Country"].value_counts())
print(df["Country"].unique())
print()


print("Description: ")
contagem_produtos = df['Description'].value_counts().reset_index()
contagem_produtos = ['Description', 'Count']
print(contagem_produtos)
print()


#### Pré-processamento ####


# Verificando se há valorea NA
print(df.isnull().sum()) # Valores NA presentes nas variáveis "Description" e "CustomerID"


# Para este estudo de caso, vou eliminar as observações que apresentam valores NA
print("Shape após remover valores ausêntes: ")
df = df.dropna()
print(df.shape) # 406829 x 8
print()


# Categorizando a variável "Description"
print("Tipos de variáveis após a converção da variável 'Description' para o tipo category: ")
df['Description'] = df['Description'].astype('category')
print(df.dtypes)
print()


# Filtrando as linhas em que a quantidade de produtos vendidos é maior que zero, descartando as linhas em que a
# # quantidade é zero ou negativa
df = df[df['Quantity'] > 0]


# Como a maioria das vendas foram feitas no Reino unido, a análise será filtrada apenas para este país
df = df[df['Country'] == 'United Kingdom']


# Transformação dos dados em matriz binária

# Primeira Etapa: agrupar os dados por InvoiceNo e Description, agrupar por transação e item comprado.
# Somando a quantidade de itens comprados em cada transação.

# Segunda Etapa: O método unstack() para pivotar os dados de forma que cada item comprado seja uma coluna e cada transação
# seja uma linha.

# Terceira Etapa: O método set_index() para definir InvoiceNo como o índice do dataframe, onde, cada linha do dataframe
# representa uma transação e está identificada pelo número da fatura.

# Quarta Etapa: Substituíndo os valores não nulos por 1 e os valores nulos por 0 usando o método fillna(0).

# O resultado final é uma matriz binária onde cada linha representa uma transação e cada coluna representa um item
# comprado, com valores 1 ou 0 indicando se o item foi ou não comprado em cada transação.

cesto = (df.groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))


#  Função para transformar os valores da matriz em uma representação binária, o que é necessário para a análise de
#  regras de associação.
def valores_binarios(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


# Utilizand método applymap é usado para aplicar a função valores_binarios a cada elemento da matriz basket.
cesto_compras = cesto.applymap(valores_binarios)


# Convertendo o DataFrame em um DataFrame com tipo de dados booleanos
# Devido o pacote mlxtendPara, vou converter o df para o tipo de dados booleano para evitar que o desempenho do cálculo
# pode ser prejudicado.
basket_sets = cesto_compras.astype(bool)


# Aplicandp o algoritmo Apriori
# min_support=0.05 define a frequência mínima de suporte, que representa a porcentagem mínima de transações que contêm
# um conjunto específico de itens, para que o itemset seja considerado frequente. No caso do código, estamos definindo
# que um itemset será considerado frequente se aparecer em pelo menos 3% das transações.
itens_frequentes = apriori(basket_sets, min_support=0.03, use_colnames=True)


# Gerando as regras de associação
# metric: a métrica usada para avaliar a força das regras de associação. A métrica "lift",  que mede a força de
# associação entre os antecedentes e consequentes, levando em consideração a frequência dos  antecedentes e consequentes.
# min_threshold: valor mínimo de métrica de associação necessário para que uma regra seja considerada relevante.
regras = association_rules(itens_frequentes, metric="lift", min_threshold=3)

# exibição das regras de associação e suas métricas
# "antecedents": os itens antecedentes da regra, ou seja, os itens que apareceram antes na transação e que, por isso,
# são usados para prever a ocorrência do item consequente.
# "consequents": os itens consequentes da regra, ou seja, os itens que apareceram depois na transação e que são
# previstos a partir dos itens antecedentes.
# "support": a frequência de ocorrência conjunta dos itens antecedentes e consequentes na base de dados.
# "confidence": a proporção de transações que incluem todos os itens antecedentes e consequentes em relação ao total de
# transações que incluem os itens antecedentes.
# "lift": a medida da força da associação entre os itens antecedentes e consequentes, comparando a frequência observada
# com a frequência esperada caso os itens fossem independentes
print(regras[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#                  antecedents                consequents  ...  confidence      lift
# 0  (JUMBO BAG RED RETROSPOT)  (JUMBO BAG PINK POLKADOT)  ...    0.349689  7.169917
# 1  (JUMBO BAG PINK POLKADOT)  (JUMBO BAG RED RETROSPOT)  ...    0.623153  7.169917




