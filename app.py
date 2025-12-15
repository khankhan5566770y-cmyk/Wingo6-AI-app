import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model

   # Model load karo
   model = load_model("mock_model.h5")

   # Streamlit UI
   st.title("AI Prediction App")
   st.write("Enter 5 values to get prediction")

   inputs = []
   for i in range(5):
       val = st.number_input(f"Feature {i+1}", value=0.0)
       inputs.append(val)

   if st.button("Predict"):
       data = np.array([inputs])
       prediction = model.predict(data)
       predicted_class = np.argmax(prediction)
       st.write("Prediction probabilities:", prediction)
       st.write("Predicted class:", predicted_class)
