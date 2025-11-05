import streamlit as st
import joblib
import numpy as np
from sklearn.datasets import load_iris

# Load model
model = joblib.load("random_forest_model.pkl")

# Load iris dataset (for target names)
iris = load_iris()
target_names = iris.target_names

# Streamlit app setup
st.set_page_config(page_title="Iris Classifier", page_icon="🌸")
st.title("🌸 Iris Flower Classification App")
st.write("This Streamlit app uses a **Random Forest model** trained and logged via **MLflow on DagsHub**.")
st.markdown("[View MLflow Experiment →](https://dagshub.com/malaychand/daghub-connect.mlflow/)")

# User inputs
st.header("🔧 Input Flower Measurements")
sepal_length = st.slider("Sepal length (cm)", 4.0, 8.0, 5.1)
sepal_width = st.slider("Sepal width (cm)", 2.0, 4.5, 3.5)
petal_length = st.slider("Petal length (cm)", 1.0, 7.0, 1.4)
petal_width = st.slider("Petal width (cm)", 0.1, 2.5, 0.2)

# Predict
if st.button("🌼 Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"🌸 Predicted Iris Species: **{target_names[prediction]}**")

st.markdown("---")
st.caption("Built with ❤️ by Malay Chand | Powered by Streamlit & DagsHub MLflow")
