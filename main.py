# Trabalho de Análise de Dados em Saúde Pública
# Intregantes: Bruno Trevizan Rizzatto, Gustavo Rossi Silva, Yuji Chikara Kiyota

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from colorama import Fore, Style
from tabulate import tabulate


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


# Carregar o dataset
df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Preencher os valores nulos de 'bmi' com a média
df['bmi'] = df['bmi'].fillna(df['bmi'].mean())

graficos_de_risco(df)

# Converter variáveis categóricas para numéricas usando one-hot encoding
df = pd.get_dummies(df, drop_first=True)

# Definir as features (X) e o target (y)
X = df.drop(columns=['stroke'])
y = df['stroke']

# Balancear os dados usando SMOTE no conjunto de treino
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Dividir os dados balanceados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Padronizar os dados
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Usar Random Forest e otimizar hiperparâmetros com Grid Search
rf = RandomForestClassifier(class_weight='balanced', random_state=42)

# Definir hiperparâmetros para otimização
param_grid = {
    'n_estimators': [100, 150, 200],
    'max_depth': [None, 30],
    'min_samples_split': [2],
    'min_samples_leaf': [1]
}

# Usar Grid Search para encontrar os melhores hiperparâmetros
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, scoring='f1', cv=5)
grid_search.fit(X_train, y_train)

# Melhor modelo encontrado
best_rf = grid_search.best_estimator_

# Fazer previsões no conjunto de teste
y_pred = best_rf.predict(X_test)

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, target_names=['Sem AVC', 'Com AVC'], output_dict=True, zero_division=0)
cross_val_score_mean = cross_val_score(best_rf, X, y, cv=5, scoring='accuracy').mean()

# Exibindo resultados

print(Fore.RED + "\nPrecisão no conjunto de teste: " + Fore.YELLOW + f"{accuracy:.2f}" + Style.RESET_ALL)

print(Fore.WHITE + "\nMatriz de Confusão:\n")
conf_matrix_table = tabulate(conf_matrix, headers=['Sem AVC', 'Com AVC'], tablefmt="fancy_grid", showindex=['Sem AVC', 'Com AVC'])
print(Fore.WHITE + conf_matrix_table + Style.RESET_ALL)

print(Fore.RED + "\nRelatório de Classificação (Random Forest):\n")
print(Fore.WHITE + f"{'Classe':<15}{'Precisão':<10}{'Recall':<10}{'F1-Score':<10}{'Suporte':<10}")
for rotulo, metrics in class_report.items():
    if rotulo == 'accuracy':
        print(f"{rotulo:<15}{metrics:<10.2f}")
    else:
        print(f"{rotulo:<15}{metrics['precision']:<10.2f}{metrics['recall']:<10.2f}{metrics['f1-score']:<10.2f}{metrics['support']:<10}")

print(Fore.RED + "\nMédia de Validação Cruzada (Cross-Validation): " + Fore.YELLOW + f"{cross_val_score_mean:.2f}" + Style.RESET_ALL)