import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import wilcoxon
import warnings

# Ignorar warnings de convergencia para mantener la salida limpia en este ejemplo
warnings.filterwarnings('ignore')

# ==========================================
# 1. CARGA DE DATOS (T1 - Inicio)
# ==========================================
# SUSTITUYE ESTO POR TUS DATOS DE LA PRÁCTICA 2
# data = pd.read_csv('tu_archivo.csv')
# X_original = data.drop('target', axis=1).values
# y = data['target'].values

# Ejemplo con Breast Cancer (para que el código sea ejecutable ahora mismo)
data_dummy = load_breast_cancer()
X_original = data_dummy.data
y = data_dummy.target

# Estandarizar es crucial para PCA y Redes Neuronales
scaler = StandardScaler()
X_original = scaler.fit_transform(X_original)

print(f"Datos Originales shape: {X_original.shape}")

# ==========================================
# 2. REDUCCIÓN DE DIMENSIONALIDAD (T1)
# ==========================================

# --- A) PCA (Mínimo 60% varianza) ---
pca = PCA(n_components=0.60) # 0.60 significa que elegirá componentes para el 60% varianza
X_pca = pca.fit_transform(X_original)
print(f"Datos PCA shape: {X_pca.shape} (Varianza explicada: {np.sum(pca.explained_variance_ratio_):.2f})")

# --- B) Autoencoder (MLPRegressor) ---
# Objetivo: Reducir a la mitad de características
n_features = X_original.shape[1]
n_hidden = int(n_features / 2)

print(f"Entrenando Autoencoder para reducir a {n_hidden} features...")

# Búsqueda de hiperparámetros para el Autoencoder (T1.2)
# El Autoencoder aprende la función Identidad: Entrada X -> Salida X
ae_params = {
    'hidden_layer_sizes': [(n_hidden,)], # La capa oculta es el cuello de botella
    'activation': ['relu', 'tanh'],
    'max_iter': [200, 500]
}

ae_grid = GridSearchCV(
    MLPRegressor(random_state=42),
    ae_params,
    cv=3,
    scoring='neg_mean_squared_error' # Queremos minimizar el error de reconstrucción
)
ae_grid.fit(X_original, X_original) # X vs X

print(f"Mejor Autoencoder: {ae_grid.best_params_}")
best_ae = ae_grid.best_estimator_

# TRUCO: MLPRegressor no tiene método 'transform' para darnos la capa oculta.
# Tenemos que calcular manualmente la salida de la capa oculta.
def get_hidden_layer_output(model, X):
    # W1 son los pesos de entrada -> oculta
    # b1 son los bias de la oculta
    W1 = model.coefs_[0]
    b1 = model.intercepts_[0]
    
    # Operación lineal
    z = np.dot(X, W1) + b1
    
    # Aplicar función de activación
    if model.activation == 'relu':
        return np.maximum(0, z)
    elif model.activation == 'tanh':
        return np.tanh(z)
    elif model.activation == 'logistic':
        return 1 / (1 + np.exp(-z))
    else: # identity
        return z

X_ae = get_hidden_layer_output(best_ae, X_original)
print(f"Datos Autoencoder shape: {X_ae.shape}")


# ==========================================
# 3. PREPARACIÓN DE TODOS LOS DATASETS (T2)
# ==========================================
# Aquí meterías los datasets de tus compañeros si los tuvieras.
datasets = {
    "Original": X_original,
    "PCA": X_pca,
    "Autoencoder": X_ae
    # "Compañero1_Original": ...
}

# ==========================================
# 4. CLASIFICACIÓN Y VALIDACIÓN (T3)
# ==========================================

# Diccionario para guardar resultados para el test de Wilcoxon (T4)
# Estructura: results_store[nombre_dataset][nombre_clasificador] = [score_fold1, score_fold2, ...]
results_store = {name: {'kNN': [], 'MLP': []} for name in datasets.keys()}

# Parámetros a optimizar (GridSearch interno)
knn_params = {'n_neighbors': [3, 5, 7, 9]} # T3.1
mlp_params = { # T3.2
    'hidden_layer_sizes': [(10,), (50,), (100,)],
    'activation': ['relu', 'tanh'],
    'max_iter': [200, 500]
}

# Validación cruzada externa (para evaluar)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n--- Iniciando Evaluación de Clasificadores ---")

for data_name, X_current in datasets.items():
    print(f"\nProcesando dataset: {data_name}")
    
    # Loop de Validación Cruzada Externa
    fold_idx = 0
    for train_idx, test_idx in outer_cv.split(X_current, y):
        X_train, X_test = X_current[train_idx], X_current[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 1. k-NN
        # GridSearchCV hace la validación interna (train/val) automáticamente
        knn_grid = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3)
        knn_grid.fit(X_train, y_train)
        acc_knn = knn_grid.score(X_test, y_test)
        results_store[data_name]['kNN'].append(acc_knn)
        
        # 2. MLP Classifier
        mlp_grid = GridSearchCV(MLPClassifier(random_state=42), mlp_params, cv=3)
        mlp_grid.fit(X_train, y_train)
        acc_mlp = mlp_grid.score(X_test, y_test)
        results_store[data_name]['MLP'].append(acc_mlp)
        
        fold_idx += 1

# ==========================================
# 5. REPORTE DE RESULTADOS (T4)
# ==========================================
print("\n--- Resultados Promedio (Mean +/- Std) ---")
for data_name in results_store:
    for clf_name in results_store[data_name]:
        scores = results_store[data_name][clf_name]
        print(f"{data_name} + {clf_name}: {np.mean(scores):.4f} +/- {np.std(scores):.4f}")

# ==========================================
# 6. ANÁLISIS ESTADÍSTICO: WILCOXON (T5)
# ==========================================
print("\n--- Test de Wilcoxon ---")

# Ejemplo 1: Comparar Original vs PCA usando kNN
scores_original_knn = results_store['Original']['kNN']
scores_pca_knn = results_store['PCA']['kNN']

# Wilcoxon requiere arrays de diferencias
if len(scores_original_knn) == len(scores_pca_knn):
    stat, p_value = wilcoxon(scores_original_knn, scores_pca_knn, alternative='two-sided')
    print(f"Original vs PCA (kNN): p-value = {p_value:.4f}")
    if p_value < 0.05:
        print("-> Hay diferencia estadísticamente significativa.")
    else:
        print("-> No hay diferencia significativa.")

# Ejemplo 2: Comparar kNN vs MLP en dataset Original
scores_original_mlp = results_store['Original']['MLP']
stat, p_value = wilcoxon(scores_original_knn, scores_original_mlp, alternative='two-sided')
print(f"kNN vs MLP (Original): p-value = {p_value:.4f}")

# NOTA: Debes hacer estas comparaciones para todas las combinaciones que te interesen en el reporte.
