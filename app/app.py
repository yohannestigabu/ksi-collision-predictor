import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# ------------------------------------------------------------
# Streamlit App Configuration
# ------------------------------------------------------------
st.set_page_config(page_title="KSI Collision Predictor", page_icon="🚗", layout="centered")
st.title("🚗 KSI Collision Severity Predictor")
st.write("Enter conditions and get a prediction: **Fatal** vs **Non-Fatal Injury**.")


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------
def categorize_age(X):
    def age_category(age_range):
        if age_range in ['0 to 4', '5 to 9', '10 to 14']:
            return 'kid'
        elif age_range == '15 to 19':
            return 'teenager'
        elif age_range in ['20 to 24', '25 to 29']:
            return 'youth'
        elif age_range in ['30 to 34', '35 to 39', '40 to 44', '45 to 49',
                           '50 to 54', '55 to 59', '60 to 64']:
            return 'adult'
        elif age_range in ['65 to 69', '70 to 74', '75 to 79', '80 to 84',
                           '85 to 89', '90 to 94', 'Over 95']:
            return 'old'
        else:
            return 'unknown'

    transformed_column = X.iloc[:, 0].apply(age_category)
    return pd.DataFrame(transformed_column, columns=[X.columns[0]])


def frequency_encode(X):
    freq = X.iloc[:, 0].value_counts(normalize=True)
    transformed_column = X.iloc[:, 0].map(freq)
    return pd.DataFrame(transformed_column, columns=[X.columns[0]])


def binary_transform_array(X):
    """
    Converts specific columns with 'Yes'/'No' values to 1/0 in a numpy array.
    """
    yes_no_indices = [12,13,14,15,16,17,18,19,20,21,22,25,26,27,28]
    X = np.array(X, dtype=object)
    for idx in yes_no_indices:
        if idx < X.shape[1]:
            X[:, idx] = np.where(X[:, idx] == 'Yes', 1, 0)
    return X


# ------------------------------------------------------------
# Load model and input column list
# ------------------------------------------------------------
MODEL_PATH = "app/ksi_model.pkl"
COLS_PATH = "app/input_columns.json"

try:
    model = joblib.load(MODEL_PATH)
    with open(COLS_PATH) as f:
        expected_cols = json.load(f)
except Exception as e:
    st.error(f"❌ Failed to load model or column schema: {e}")
    st.stop()


# ------------------------------------------------------------
# User Interface
# ------------------------------------------------------------
col1, col2 = st.columns(2)
with col1:
    road_class = st.selectbox("Road Class", ["Arterial", "Collector", "Local"])
    light = st.selectbox("Light Condition", [
        "Daylight", "Dark", "Dark, artificial", "Dusk",
        "Dawn, artificial", "Dawn", "Dusk, artificial",
        "Daylight, artificial", "Other"
    ])
    visibility = st.selectbox("Visibility", [
        "Clear", "Rain", "Snow", "Fog, Mist, Smoke, Dust",
        "Strong wind", "Other", "Drifting Snow", "Freezing Rain"
    ])
    rdsfcond = st.selectbox("Road Surface", [
        "Dry", "Wet", "Loose Snow", "Ice", "Slush",
        "Packed Snow", "Spilled liquid", "Loose Sand or Gravel", "Other"
    ])
with col2:
    injury = st.selectbox("Injury Severity", [
        "Minimal", "Minor", "Major", "None", "Other Property Owner"
    ])
    invtype = st.selectbox("Involved Type", [
        "Driver", "Pedestrian", "Cyclist", "Passenger", "Vehicle Owner",
        "Truck Driver", "Other Property Owner", "Other"
    ])
    district = st.selectbox("District", [
        "Toronto and East York", "Scarborough", "Etobicoke York", "North York"
    ])
    loccoord = st.selectbox("Location Type", [
        "Intersection", "Non-Intersection", "Other"
    ])
    trafcctl = st.selectbox("Traffic Control", [
        "No Control", "Traffic Controller", "Traffic Signal",
        "Stop Sign", "Yield Sign", "Other"
    ])


# ------------------------------------------------------------
# Build Input DataFrame
# ------------------------------------------------------------
row = pd.Series({c: None for c in expected_cols})

# Map UI selections to model input columns
set_map = {
    "ROAD_CLASS": road_class,
    "LIGHT": light,
    "VISIBILITY": visibility,
    "RDSFCOND": rdsfcond,
    "INJURY": injury,
    "INVTYPE": invtype,
    "DISTRICT": district,
    "LOCCOORD": loccoord,
    "TRAFFCTL": trafcctl,
}

for k, v in set_map.items():
    if k in row.index:
        row[k] = v

# Set defaults for binary flags
for b in ["PEDESTRIAN", "CYCLIST", "AUTOMOBILE", "MOTORCYCLE",
          "TRUCK", "TRSN_CITY_VEH", "PASSENGER", "SPEEDING",
          "AG_DRIV", "REDLIGHT", "ALCOHOL"]:
    if b in row.index and pd.isna(row[b]):
        row[b] = "No"

# Set reasonable defaults for other features
if "INVAGE" in row.index and pd.isna(row["INVAGE"]):
    row["INVAGE"] = "30 to 34"
if "NEIGHBOURHOOD_158" in row.index and pd.isna(row["NEIGHBOURHOOD_158"]):
    row["NEIGHBOURHOOD_158"] = "Unknown"

X_one = pd.DataFrame([row])


# ------------------------------------------------------------
# Prediction Button
# ------------------------------------------------------------
st.divider()
if st.button("🔮 Predict"):
    try:
        # Make prediction
        y_pred = model.predict(X_one)[0]
        proba = model.predict_proba(X_one)[0]
        fatal_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])

        # Clean label formatting
        
                # Clean label formatting
        try:
            # Convert numpy types or encoded labels to string safely
            label = str(y_pred).strip()

            # If it's numeric, map manually
            if label in ["0", "1"]:
                label = "Fatal" if label == "1" else "Non-Fatal Injury"

            # Clean up any brackets or array formatting
            label = label.replace("[", "").replace("]", "").replace("'", "")

            # Handle any stray encodings
            if "Fatal" not in label and "Non-Fatal" not in label:
                label = "Non-Fatal Injury" if fatal_prob < 0.5 else "Fatal"

        except Exception:
            label = "Non-Fatal Injury"
        
        # Display result
        st.subheader(f"Prediction: **{label}**")
        st.write(f"Confidence (Fatal): **{fatal_prob * 100:.1f}%**")
        
        if "Fatal" in label and fatal_prob >= 0.5:
            st.error("⚠️ Higher severity predicted (Fatal collision likely).")
        else:
            st.success("✅ Lower severity predicted (Non-Fatal Injury).")

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.caption("Tip: make sure the saved model and input_columns.json match your trained notebook version.")