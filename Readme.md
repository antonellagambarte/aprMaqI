# README - Trabajo Práctico Final: Predicción de Accidentes Cerebrovasculares

Este proyecto corresponde al trabajo práctico final de la materia Aprendizaje de Máquina (CEIA, FIUBA). El objetivo principal es analizar y comparar distintos modelos de machine learning para predecir la ocurrencia de accidentes cerebrovasculares (ACV) utilizando el [Stroke Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/stroke-prediction-dataset).

## Estructura del archivo `tpfinal.ipynb`

El notebook está organizado en las siguientes secciones:

### 1. Presentación y Objetivo
- Introducción al problema y justificación del análisis.
- Descripción del dataset y enlace a la fuente.

### 2. Análisis Exploratorio de Datos (EDA)
- Carga de datos y revisión de estructura.
- Análisis de duplicados y valores nulos.
- Visualización de distribuciones y outliers.
- Estudio de correlaciones entre variables numéricas y categóricas.

### 3. Preprocesamiento y Preparación
- Imputación de valores nulos en BMI según grupos de edad.
- Transformación logarítmica de BMI para mitigar outliers.
- Codificación de variables categóricas con One-Hot Encoding.
- Agrupación de categorías poco frecuentes (ej. género "Other" y tipo de trabajo "Never_worked").
- Escalado de variables numéricas.
- División en conjuntos de entrenamiento y test.
- Aplicación de técnicas de balanceo (SMOTE y RandomOverSampler).

### 4. Modelización
- Implementación desde cero de un árbol de clasificación (`ArbolClasificacion` y `Nodo`).
- Entrenamiento y evaluación de los siguientes modelos:
  - Árbol de Decisión (con y sin balanceo)
  - SVM (con y sin balanceo)
  - Random Forest
  - XGBoost (con y sin balanceo)
- Optimización de hiperparámetros con Optuna.
- Visualización preliminar de métricas y curvas ROC.

### 5. Evaluación de Resultados
- Comparación cuantitativa de accuracy, precision, recall y f1-score entre modelos.
- Visualización conjunta de curvas ROC.
- Conclusión:
    - Resumen de hallazgos principales.
    - Recomendaciones sobre el uso de modelos en contextos desbalanceados.

## Decisiones de Modelado y Preprocesamiento

- **Imputación de BMI**: Se realiza por mediana en grupos etarios para preservar la relación entre edad y masa corporal.
- **Transformación de BMI**: Se aplica logaritmo para reducir el efecto de outliers.
- **Codificación de variables**: Se utiliza One-Hot Encoding para todas las variables categóricas relevantes.
- **Agrupación de categorías raras**: Género "Other" se fusiona con "Female"; "Never_worked" se agrupa con "Children" en tipo de trabajo.
- **Balanceo de clases**: Se emplean SMOTE y RandomOverSampler para mejorar la sensibilidad de los modelos hacia la clase minoritaria (ACV).
- **Optimización**: Se utiliza Optuna para encontrar los mejores hiperparámetros en cada modelo.
- **Evaluación**: Se prioriza el recall sobre la precisión para la clase positiva, dada la importancia clínica de no omitir casos de ACV.

## Requisitos

- Python 3.11+
- Bibliotecas: pandas, numpy, matplotlib, seaborn, scikit-learn, imblearn, optuna, xgboost

Instalación recomendada (ver primeras celdas del notebook):

```python
pip install pandas matplotlib seaborn scikit-learn numPy 
pip install imblearn
pip install optuna
pip install --force-reinstall xgboost