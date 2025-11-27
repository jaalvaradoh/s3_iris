import streamlit as st
import joblib

# ============================
# Cargar modelos Iris
# ============================
knn_model = joblib.load('./model/modelo_iris_knn.pkl')
svm_model = joblib.load('./model/modelo_iris_svm.pkl')
tree_model = joblib.load('./model/modelo_iris_arbol.pkl')

# ============================
# Interfaz
# ============================
st.title("Predicción del dataset Iris")
st.write("Ingresa las características de la flor:")

# Campos de entrada numéricos
sepal_length = st.number_input('sepal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input('sepal width (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input('petal length (cm)', min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input('petal width (cm)', min_value=0.0, max_value=10.0, step=0.1)

# Selección de modelos
st.write("Selecciona los modelos que deseas usar para la predicción:")
use_knn = st.checkbox("KNN")
use_svm = st.checkbox("SVM")
use_tree = st.checkbox("Árbol de Decisión")

# Botón de predicción
if st.button("Predecir"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    if not (use_knn or use_svm or use_tree):
        st.warning("Selecciona al menos un modelo.")
    else:
        st.write("### Resultados de la predicción")

        # KNN
        if use_knn:
            pred_knn = knn_model.predict(features)[0]
            st.write(f"**KNN Predicción:** {pred_knn}")

        # SVM
        if use_svm:
            pred_svm = svm_model.predict(features)[0]
            st.write(f"**SVM Predicción:** {pred_svm}")

        # Árbol de decisión
        if use_tree:
            pred_tree = tree_model.predict(features)[0]
            st.write(f"**Árbol de Decisión Predicción:** {pred_tree}")
