import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE

# Ler o arquivo csv
df = pd.read_csv("healthcare-dataset-stroke-data.csv")

# Preencher os valores nulos de 'bmi' com a média (correção da atribuição)
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

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

# Define features e a variável alvo para o modelo
X = df.drop(columns=['id', 'stroke'])
y = df['stroke']

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Aplicar SMOTE para balancear a classe minoritária
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Padronizar os dados
scaler = StandardScaler()
X_train_smote = scaler.fit_transform(X_train_smote)
X_test = scaler.transform(X_test)

# Treinar o modelo de Regressão Logística
model = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model.fit(X_train_smote, y_train_smote)

# Fazer previsões no conjunto de teste
y_pred = model.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Sem AVC', 'Com AVC'])
cross_val_score = cross_val_score(model, X, y, cv=5, scoring='accuracy')
print(f"Precisão: {accuracy:.2f}")
print(f"\nMatriz de Confusão:\n{conf_matrix}")
print(f"\nRelatório de Classificação:\n{class_report}")
print(f"\nCross Validation Score: {cross_val_score.mean():.2f}")

