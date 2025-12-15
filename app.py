import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("mock_model.h5")

st.title("Big or Small Prediction App")

# Input features
features = []
for i in range(5):
    val = st.number_input(f"Feature {i+1}", 0.0, 1.0, 0.5)
    features.append(val)

if st.button("Predict"):
    data = np.array([features])
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction, axis=1)

    # Map 0 -> Small, 1 -> Big
    label_map = {0: "Small", 1: "Big"}
    result = label_map[int(predicted_class[0])]

    st.write("Prediction probabilities:", prediction)
    st.write("Predicted class:", result)
