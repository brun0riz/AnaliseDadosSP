# Analise de dados de saude publica
# Integrantes do grupo: Bruno Trevizan, Gustavo Rossi, Yuji Kiyota

#  - Modelagem Simples: Implementar uma classificação binária simples para prever a
# probabilidade de um paciente sofrer um AVC.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Ler o arquivo csv
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Verificar os valores nulos

# print(df.isnull().sum())
# Apenas os os valores de 'bmi' que estão nulos

# Preencher os valores nulos de 'bmi' com a média
df['bmi'].fillna(df['bmi'].mean(), inplace=True)

# Gráficos para fatores de risco para AVC

# Com problemas cardiacos
sns.countplot(x='heart_disease', data=df, hue='stroke')
plt.title('Avc por problemas cardiacos')
plt.xlabel('Problemas cardiacos')
plt.ylabel('Quantidade de pessoas')
plt.show()

# Com problemas de hipertensão
sns.countplot(x='hypertension', data=df, hue='stroke')
plt.title('Avc por hipertensão')
plt.xlabel('Hipertensão')
plt.ylabel('Quantidade de pessoas')
plt.show()

# Com problemas nivel de glicose
sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
plt.title('Nivel de glicose por AVC')
plt.xlabel('AVC')
plt.ylabel('Nivel de glicose')
plt.show()

# bmi
sns.boxplot(x='stroke', y='bmi', data=df)
plt.title('BMI por AVC')
plt.xlabel('AVC')
plt.ylabel('BMI')
plt.show()

# Fumante
sns.countplot(x='smoking_status', data=df, hue='stroke')
plt.title('Avc por fumante')
plt.xlabel('Fumantes')
plt.ylabel('Quantidade de pessoas')
plt.show()


# Histograma de Distribuição de Idades
plt.hist(df['age'], bins=20, color='blue', edgecolor='black')
plt.title('Distribuição de Idades')
plt.xlabel('Idade')
plt.ylabel('Quantidade de pessoas')
plt.show()

# Modelagem Simples

# Transformar as variáveis categóricas em numéricas

label_encoders = {}
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features e a varivel alvo para o modelo
X = df.drop(columns=['id', 'stroke'])  # X vai conter todas as colunas exceto 'id' e 'stroke'
# isso ocorre pois a coluna 'id' não é relevante para o modelo e a coluna 'stroke' é a nossa variável alvo
y = df['stroke']  # y vai conter apenas a coluna 'stroke', que seria a nossa varivel alvo

# Dividir os dados em treino e teste, separando em 80% para treino e 20% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Treinar o modelo, utilizando Regressão Logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Aqui o modelo faz previsões com base nos dados de teste de X_test e armazena em y_pred
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)  # é calculada comparando as previões com os valores reais de y_test
conf_matrix = confusion_matrix(y_test, y_pred)  # mostra o número de acertos e erros do modelo
class_report = classification_report(y_test, y_pred)  # Fonerce um relatório com as métricas de precisão, recall e f1-score

print(f"Acurácia: {accuracy}")
print("Matriz de Confusão:")
print(conf_matrix)
print("Relatório de Classificação:")
print(class_report)




