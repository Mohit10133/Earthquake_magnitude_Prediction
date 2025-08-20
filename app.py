import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the trained model
with open("earthquake_xgb_mag.pkl", "rb") as f:
    model = pickle.load(f)


# Load dataset for encoding references
file_path = "Earthquake.csv"
df = pd.read_csv(file_path)

# Encode categorical features using the same encoding as during training
le_alert = LabelEncoder()
df['alert'] = le_alert.fit_transform(df['alert'])
le_magType = LabelEncoder()
df['magType'] = le_magType.fit_transform(df['magType'])

# Standard scaler for numerical features
scaler = StandardScaler()
numerical_features = ['cdi', 'mmi', 'sig', 'alert', 'nst', 'dmin', 'gap', 'magType', 'depth', 'latitude', 'longitude']
scaler.fit(df[numerical_features])

# Streamlit UI
st.title("🌍 Earthquake Magnitude Prediction")

st.markdown("""
### Enter Earthquake Parameters
Fill in the details below to predict the earthquake magnitude.
""")

# Input fields for user
latitude = st.number_input("Latitude", value=0.0, format="%.6f")
longitude = st.number_input("Longitude", value=0.0, format="%.6f")
depth = st.number_input("Depth (km)", value=0.0, format="%.2f")
nst = st.number_input("Number of Stations (nst)", value=0, min_value=0)
gap = st.number_input("Gap (degrees)", value=0.0, format="%.2f")
dmin = st.number_input("Minimum Distance (dmin)", value=0.0, format="%.4f")
alert = st.selectbox("Alert Level", options=le_alert.classes_)
magType = st.selectbox("Magnitude Type", options=le_magType.classes_)
cdi = st.number_input("CDI", value=0.0, format="%.4f")
mmi = st.number_input("MMI", value=0.0, format="%.4f")
sig = st.number_input("Significant", value=0.0, format="%.4f")

# Encode categorical inputs
alert_encoded = le_alert.transform([alert])[0]
magType_encoded = le_magType.transform([magType])[0]

# Prepare input features
input_data = np.array([[latitude, longitude, depth, nst, gap, sig, dmin, cdi, mmi, alert_encoded, magType_encoded]])

# Scale numerical features
input_data[:, :len(numerical_features)] = scaler.transform(input_data[:, :len(numerical_features)])

# Prediction
if st.button("Predict Magnitude"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Earthquake Magnitude: **{prediction:.2f}**")
