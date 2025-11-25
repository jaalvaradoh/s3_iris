import streamlit as st
import joblib
import numpy as np

st.title("Clasificación Iris con 3 Modelos (KNN, Árbol, SVM)")
st.write("0 = setosa, 1 = versicolor, 2 = virginica")

# -------------------------------------------------------------
# Cargar modelos desde archivos locales o GitHub (raw)
# -------------------------------------------------------------

#@st.cache_resource
def cargar_modelos():
    knn = joblib.load('./modelo_iris_knn.pkl')
    arbol = joblib.load('./modelo_iris_knn.pkl')
    svm = joblib.load('./modelo_iris_knn.pkl')
    return knn, arbol, svm

knn_model, arbol_model, svm_model = cargar_modelos()

# -------------------------------------------------------------
# Entradas del usuario
# -------------------------------------------------------------
st.subheader("Ingresa los valores:")

sepal_length = st.number_input("Sepal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width  = st.number_input("Sepal width (cm)",  min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width  = st.number_input("Petal width (cm)",  min_value=0.0, max_value=10.0, step=0.1)

X = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

# -------------------------------------------------------------
# Botones para predecir
# -------------------------------------------------------------

if st.button("Predecir con KNN"):
    pred = knn_model.predict(X)[0]
    st.success(f"Predicción KNN: {pred}")

if st.button("Predecir con Árbol de Decisión"):
    pred = arbol_model.predict(X)[0]
    st.success(f"Predicción Árbol: {pred}")

if st.button("Predecir con SVM"):
    pred = svm_model.predict(X)[0]
    st.success(f"Predicción SVM: {pred}")
