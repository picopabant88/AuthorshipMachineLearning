import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_predict
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from tabulate import tabulate
from gensim.models import Word2Vec

# Descargar recursos necesarios de NLTK
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Cargar datos
archivo_csv = "archivo_utf8.csv"
df = pd.read_csv(archivo_csv)

if df.shape[1] != 4:
    raise ValueError("CSV file have to cotain 4 columns.")

textos = df.iloc[:, 2].tolist()
etiquetas = df.iloc[:, 3].tolist()

# Preprocesamiento de texto
def preprocesar_texto(texto):
    stop_words = set(stopwords.words('spanish'))
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(texto.lower())
    tokens = [token for token in tokens if token.isalpha()]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return " ".join(tokens)

# Vectorización TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=1000)
textos_limpios = [preprocesar_texto(texto) for texto in textos]
X_tfidf = tfidf_vectorizer.fit_transform(textos_limpios)

# Modelo Word Embedding
model_w2v = Word2Vec(sentences=[word_tokenize(texto) for texto in textos_limpios], vector_size=100, window=5, min_count=1, sg=0)

def obtener_vector_promedio(texto, model):
    vectors = [model.wv[word] for word in word_tokenize(texto) if word in model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    return np.zeros(model.vector_size)

X_w2v = np.array([obtener_vector_promedio(texto, model_w2v) for texto in textos_limpios])
X_combined = np.hstack((X_tfidf.toarray(), X_w2v))

# Escalar características
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X_combined)

# Codificar etiquetas
label_encoder = LabelEncoder()
etiquetas_encoded = label_encoder.fit_transform(etiquetas)

# Definir modelos e hiperparámetros
hiperparametros = {
    "SVC Lineal": {"modelo": SVC(probability=True), "parametros": {"C": [0.1, 1, 10, 100], "kernel": ["linear"]}},
    "SVC RBF": {"modelo": SVC(probability=True), "parametros": {"C": [0.1, 1, 10, 100], "gamma": [0.001, 0.01, 0.1, 1], "kernel": ["rbf"]}},
    "Random Forest": {"modelo": RandomForestClassifier(), "parametros": {"n_estimators": [10, 50, 100, 200], "max_depth": [None, 10, 20, 30]}},
    "Regresión Logística": {"modelo": LogisticRegression(), "parametros": {"C": [0.01, 0.1, 1, 10, 100], "solver": ["lbfgs", "liblinear"]}},
    "Árbol de Decisión": {"modelo": DecisionTreeClassifier(), "parametros": {"max_depth": [None, 5, 10, 20]}},
    "K Vecinos Más Cercanos": {"modelo": KNeighborsClassifier(), "parametros": {"n_neighbors": [3, 5, 7, 9]}},
    "Naive Bayes": {"modelo": GaussianNB(), "parametros": {}}
}

mejor_modelo = None
mejor_score = 0
mejor_nombre = ""

for nombre, config in hiperparametros.items():
    print(f"\nOptimiing hyperparameters for : {nombre}")
    modelo = GridSearchCV(config["modelo"], config["parametros"], cv=10, scoring='accuracy')
    modelo.fit(X_scaled, etiquetas_encoded)
    #if modelo.best_score_ > mejor_score:
    mejor_score = modelo.best_score_
    print(f"Best hyperparameters with: {nombre}: {modelo.best_params_}")
    print(f"Best selected model : {mejor_nombre} with score: {mejor_score:.4f}")
    mejor_modelo = modelo.best_estimator_
    mejor_nombre = nombre

    # Validación cruzada con 10 folds
    predicciones = cross_val_predict(mejor_modelo, X_scaled, etiquetas_encoded, cv=10)
    
    # 10. Evaluar con validación cruzada (10 folds) para Accuracy
    accuracy_scores = cross_val_score(mejor_modelo, X_scaled, etiquetas_encoded, cv=10, scoring='accuracy')
    print("Accuracy:", accuracy_scores)
    #precision_scores = cross_val_score(mejor_modelo, X_scaled, etiquetas_encoded, cv=10, scoring='precision_weighted')
    #print("Precision:", precision_scores)
    #recall_scores = cross_val_score(mejor_modelo, X_scaled, etiquetas_encoded, cv=10, scoring='recall_weighted')
    
    #print("Recall:", recall_scores)
    f1_scores = cross_val_score(mejor_modelo, X_scaled, etiquetas_encoded, cv=10, scoring='f1_weighted')
    print("F1-score:", f1_scores)
    
    # Generar matriz de confusión
    
    print("---Target")
    print(etiquetas_encoded)
    print("---Predict")
    print(predicciones)
    conf_matrix = confusion_matrix(etiquetas_encoded, predicciones)
    plt.figure(figsize=(6, 5))
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted researcher")
    plt.ylabel("Target researcher")
    plt.title(f"Confusion matrix - {mejor_nombre}")
    plt.show()

    # Resultados finales
    print(f"Best selected model: {mejor_nombre} with score: {mejor_score:.4f}")
