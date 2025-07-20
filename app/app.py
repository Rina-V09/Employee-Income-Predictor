import streamlit as st
import pandas as pd
import joblib

# Page config
st.set_page_config(page_title="Employee Income Prediction App", layout="wide")

# Load the model and column list
model = joblib.load("best_model.pkl")
model_columns = joblib.load("model_columns.pkl")

# Title
st.markdown("<h1 style='text-align: center;'>💼 Employee Income Prediction App</h1>", unsafe_allow_html=True)

# Create 2 unequal columns (input column narrower)
left_col, right_col = st.columns([1, 2.5])

# -------- Left Column: Inputs --------
with left_col:
    st.markdown("### 🎯 Input Employee Details")

    input_dict = {
        'age': st.slider("Age", 18, 100, 30),
        'workclass': st.selectbox("Workclass", ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay']),
        'fnlwgt': st.number_input("fnlwgt", 10000, 1000000, 200000),
        'education-num': st.slider("Education Number", 1, 20, 10),
        'marital-status': st.selectbox("Marital Status", ['Never-married', 'Married-civ-spouse', 'Divorced', 'Separated', 'Married-spouse-absent', 'Widowed']),
        'occupation': st.selectbox("Occupation", ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces']),
        'relationship': st.selectbox("Relationship", ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried']),
        'race': st.selectbox("Race", ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other']),
        'sex': st.selectbox("Sex", ['Male', 'Female']),
        'capital-gain': st.number_input("Capital Gain", 0, 100000, 0),
        'capital-loss': st.number_input("Capital Loss", 0, 100000, 0),
        'hours-per-week': st.slider("Hours per Week", 1, 100, 40),
        'native-country': st.selectbox("Native Country", ['United-States', 'India', 'Mexico', 'Philippines', 'Germany', 'Canada', 'England', 'China', 'Cuba', 'Other']),
    }

# -------- Right Column: Output --------
with right_col:
    st.markdown("### 📊 Input Summary")
    input_df = pd.DataFrame([input_dict])
    st.dataframe(input_df, use_container_width=True)

    # Preprocess input
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

    if st.button("🔮 Predict Income Class"):
        try:
            prediction = model.predict(input_encoded)[0]
            st.success(f"✅ Prediction: {prediction}")
        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
