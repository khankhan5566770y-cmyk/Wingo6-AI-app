import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# Load model (make sure mock_model.h5 aapke repo mein hai)
model = load_model("mock_model.h5")

st.title("🎲 Big vs Small Predictor")

# 1. User se last 5 results input lein (digits 0-9)
st.markdown("*Enter last 5 results (digits 0 to 9):*")
inputs = []
cols = st.columns(5)
for i in range(5):
    with cols[i]:
        val = st.number_input(f"Result {i+1}", min_value=0, max_value=9, value=0, step=1)
        inputs.append(val)

# 2. Convert digits to Big(1) or Small(0) feature
# Rule: 0-4 = Small(0), 5-9 = Big(1)
features = [1 if x >= 5 else 0 for x in inputs]

st.markdown("*Converted Features (Big=1, Small=0):*")
st.write(features)

# 3. Prediction button
if st.button("Predict"):
    data = np.array([features])
    prediction = model.predict(data)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = round(np.max(prediction)*100, 2)

    label_map = {0: "Small", 1: "Big"}
    result = label_map[predicted_class]

    st.success(f"✅ Predicted: {result}")
    st.info(f"📊 Confidence: {confidence}%")

    # 4. Save prediction history in session state
    if "history" not in st.session_state:
        st.session_state.history = []

    st.session_state.history.append({
        "inputs": inputs,
        "features": features,
        "prediction": result,
        "confidence": confidence
    })

# 5. Show prediction history
if "history" in st.session_state and st.session_state.history:
    st.markdown("### 📜 Prediction History")
    df_hist = pd.DataFrame(st.session_state.history)
    st.dataframe(df_hist)

    # 6. Frequency chart of Big vs Small in history features
    all_features = [f for h in st.session_state.history for f in h["features"]]
    freq = {
        "Small (0)": all_features.count(0),
        "Big (1)": all_features.count(1)
    }

    st.markdown("### 🧊 Big vs Small Frequency Chart")
    fig, ax = plt.subplots()
    ax.bar(freq.keys(), freq.values(), color=["skyblue", "lightgreen"])
    ax.set_ylabel("Count")
    ax.set_title("Big vs Small frequency in input features history")
    st.pyplot(fig)
