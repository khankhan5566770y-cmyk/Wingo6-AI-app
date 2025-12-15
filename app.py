import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# Load model
model = load_model("mock_model.h5")

# Prediction log file
log_file = "prediction_history.csv"

st.title("Big or Small Prediction App")

st.sidebar.header("Select Input Method")
input_method = st.sidebar.radio("Choose input method:", ("Manual", "Random"))

# Generate input features
if input_method == "Manual":
    features = []
    for i in range(5):
        val = st.number_input(f"Feature {i+1}", 0.0, 1.0, 0.5)
        features.append(val)
else:
    features = np.random.rand(5).tolist()
    st.write("Randomly Generated Features:", features)

# Prediction button
if st.button("Predict"):
    data = np.array([features])
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction)

    label_map = {0: "Small", 1: "Big"}
    result = label_map[int(predicted_class)]
    st.success(f"Predicted Class: *{result}*")
    st.write("Prediction Probabilities:", prediction)

    # Save to history
    df_entry = pd.DataFrame([features + [result]])
    df_entry.columns = [f"Feature_{i+1}" for i in range(5)] + ["Result"]

    if os.path.exists(log_file):
        old_df = pd.read_csv(log_file)
        df_entry = pd.concat([old_df, df_entry], ignore_index=True)

    df_entry.to_csv(log_file, index=False)

# Show prediction history
if os.path.exists(log_file):
    st.subheader("Prediction History")
    history_df = pd.read_csv(log_file)
    st.dataframe(history_df.tail(10))

    # Plotting count of Big vs Small
    st.subheader("Prediction Distribution")
    count = history_df["Result"].value_counts()
    fig, ax = plt.subplots()
    ax.bar(count.index, count.values, color=["skyblue", "salmon"])
    ax.set_ylabel("Count")
    ax.set_title("Big vs Small")
    st.pyplot(fig)
