# Analise de dados de saúde publica
# Integrantes do grupo: Bruno Trevizan, Gustavo Rossi, Yuji Kiyota

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


# Ler o arquivo csv
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preencher os valores nulos de 'bmi' com a média
df['bmi'].fillna(df['bmi'].mean(), inplace=True)


# Gráficos para fatores de risco para AVC
def graficos_de_risco(dataframe):
    sns.countplot(x='heart_disease', data=df, hue='stroke')
    plt.title('Avc por problemas cardiacos')
    plt.xlabel('Problemas cardiacos')
    plt.ylabel('Quantidade de pessoas')
    plt.show()

    sns.countplot(x='hypertension', data=df, hue='stroke')
    plt.title('Avc por hipertensão')
    plt.xlabel('Hipertensão')
    plt.ylabel('Quantidade de pessoas')
    plt.show()

    sns.boxplot(x='stroke', y='avg_glucose_level', data=df)
    plt.title('Nivel de glicose por AVC')
    plt.xlabel('AVC')
    plt.ylabel('Nivel de glicose')
    plt.show()

    sns.boxplot(x='stroke', y='bmi', data=df)
    plt.title('BMI por AVC')
    plt.xlabel('AVC')
    plt.ylabel('BMI')
    plt.show()

    sns.countplot(x='smoking_status', data=df, hue='stroke')
    plt.title('Avc por fumante')
    plt.xlabel('Fumantes')
    plt.ylabel('Quantidade de pessoas')
    plt.show()

    plt.hist(df['age'], bins=20, color='blue', edgecolor='black')
    plt.title('Distribuição de Idades')
    plt.xlabel('Idade')
    plt.ylabel('Quantidade de pessoas')
    plt.show()

    sns.pairplot(df, hue='stroke')
    plt.show()


graficos_de_risco(df)

# Transformar as variáveis categóricas em numéricas
label_encoders = {}
categorical_columns = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for column in categorical_columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Define features e a varivel alvo para o modelo
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Encontrar os melhores hyperparametros e treinar o modelo
model = LogisticRegression()
model.fit(X_train, y_train)

# Fazer previsões
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Sem AVC', 'Com AVC'], output_dict=True)

print(f"Precisão: {accuracy:.2f}")
print("\nMatriz de Confusão:")
print(f"Sem AVC: {conf_matrix[0]}")
print(f"Com AVC: {conf_matrix[1]}")

print("\nRelatório de Classificação:")
print(f"{'Classe':<15}{'Precisão':<10}{'Recall':<10}{'F1-Score':<10}{'Suporte':<10}")
for rotulo, metrics in class_report.items():
    if rotulo == 'accuracy':
        print(f"{rotulo:<15}{metrics:<10.2f}")
    else:
        print(f"{rotulo:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{metrics['support']:<10}")

# Explicação das variáveis do relatório de classificação
# Precisão: é a proporção de previsões corretas feitas pelo modelo
# Recall: é a proporção de verdadeiros positivos que foram identificados corretamente
# F1-Score: é a média harmônica entre precisão e recall
# Suporte: é o número de ocorrências de cada classe
# Accuracy: é a proporção de previsões corretas feitas pelo modelo
# Macro avg: é a média aritmética das métricas de precisão, recall e f1-score
# Weighted avg: é a média ponderada das métricas de precisão, recall e f1-score
# Basicamente essas duas últimas trazem um desempenho geral do modelo
# Tentei usar o Smote para balancear os dados, mas os FP aumentaram, equanto o FN e TP ficaram iguais, então optei por não usar
