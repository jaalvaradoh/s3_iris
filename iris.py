import streamlit as st
import joblib

iris_target_names = {
    0: "Setosa",
    1: "Versicolor",
    2: "Virginica"
}

#############################Pagina 1##############################    

def page1():
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
    
    st.info("Dr. Jesus Alvarado Huayhuaz")
    
    st.image("imagenes/iris_dataset.png",
                     caption="Dataset Iris")
    
    st.sidebar.image("imagenes/logo_isil_principal.jpg",
                     caption="Aplicación de Modelo de Clasificación desplegado en Streamlit")
    
    st.write("""
    El dataset Iris es uno de los conjuntos de datos más conocidos y utilizados en estadística, reconocimiento de patrones e inteligencia artificial. Fue introducido por el botánico y estadístico Ronald A. Fisher en 1936, en su artículo “The Use of Multiple Measurements in Taxonomic Problems”. Su objetivo original era demostrar el uso del análisis discriminante para clasificar especies de plantas a partir de mediciones morfológicas.
    
    El dataset contiene un total de 150 muestras de flores de iris, divididas equitativamente en tres especies:
    
    - Iris setosa
    
    - Iris versicolor
    
    - Iris virginica
    
    Para cada flor se registraron cuatro características morfométricas, medidas en centímetros:
    
    - Longitud del sépalo (sepal length)
    
    - Ancho del sépalo (sepal width)
    
    - Longitud del pétalo (petal length)
    
    - Ancho del pétalo (petal width)
    
    Estas mediciones fueron tomadas originalmente en la década de 1930 a partir de especímenes reales recolectados para estudios botánicos. Fisher utilizó este conjunto para ilustrar cómo las variables cuantitativas permiten separar estadísticamente especies a través de técnicas de clasificación.
    Actualmente, el dataset Iris es ampliamente usado para:
    
    1. Enseñanza de aprendizaje supervisado,
    
    2. Pruebas de modelos de clasificación,
    
    3. Demostraciones de técnicas estadísticas,
    
    4. Ejercicios iniciales de machine learning y visualización de datos.
    
    Su simplicidad, tamaño reducido y separabilidad parcial entre clases lo convirtieron en un estándar académico para introducir conceptos clave de la inteligencia artificial y el reconocimiento de patrones.
    """)

#############################Pagina 2##############################    

def page2():
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
                #st.write(f"**KNN Predicción:** {pred_knn}")
                st.write(f"**KNN Predicción:** {pred_knn} ({iris_target_names[pred_knn]})")
    
    
            # SVM
            if use_svm:
                pred_svm = svm_model.predict(features)[0]
                #st.write(f"**SVM Predicción:** {pred_svm}")
                st.write(f"**SVM Predicción:** {pred_svm} ({iris_target_names[pred_svm]})")
    
    
            # Árbol de decisión
            if use_tree:
                pred_tree = tree_model.predict(features)[0]
                #st.write(f"**Árbol de Decisión Predicción:** {pred_tree}")
                st.write(f"**Árbol de Decisión Predicción:** {pred_tree} ({iris_target_names[pred_tree]})")



#############################Pagina 3##############################    

def page3():
    st.write("""
    1. KNN
    2. SVM
    3. Árbol de decisión
    4. Joblib
    5. Sklearn
    6. IRIS
    """)

################################################################### 
##########################Configuracion############################    
################################################################### 

page_names_to_funcs = {
  "El dataset": page1,
  "Predicciones": page2,
  "Glosario": page3,
}

selected_page = st.sidebar.selectbox("Selecciona", page_names_to_funcs.keys())
page_names_to_funcs[selected_page]()
