import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris
import time

# -------------------------------
# Load Model & Dataset
# -------------------------------
model = joblib.load("random_forest_model.pkl")
iris = load_iris()
target_names = iris.target_names

# -------------------------------
# Streamlit Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Iris Classifier 🌸",
    page_icon="🌸",
    layout="centered",
)

# -------------------------------
# Header Section
# -------------------------------
st.title("🌸 Iris Flower Classification App")
st.markdown(
    """
This interactive app uses a **Random Forest Classifier** trained on the classic **Iris dataset**  
and logged via **MLflow**, integrated with **DagsHub** for experiment tracking.  

Explore:
- 🔗 [**DagsHub Repository**](https://dagshub.com/malaychand/daghub-connect)
- 📊 [**MLflow Tracking UI**](https://dagshub.com/malaychand/daghub-connect.mlflow/)
- 💻 [**GitHub Repository**](https://github.com/malaychand/daghub-connect)
"""
)
st.markdown("---")

# -------------------------------
# Input Section
# -------------------------------
st.header("🔧 Input Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
    petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)

with col2:
    sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
    petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# -------------------------------
# Prediction Section
# -------------------------------
if st.button("🌼 Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    predicted_class = target_names[prediction]
    
    st.success(f"🌸 **Predicted Iris Species:** {predicted_class}")
    with st.empty():
        st.write("Loading...")
        time.sleep(.5)
        st.write("Complete!")

# -------------------------------
# Footer Section
# -------------------------------
st.markdown("---")
st.caption(
    """ 
Powered by [Streamlit](https://streamlit.io), [MLflow](https://mlflow.org/), and [DagsHub](https://dagshub.com)
"""
)
