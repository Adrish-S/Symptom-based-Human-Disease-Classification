import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd

# Load the trained XGBoost model
model = xgb.XGBClassifier()
model.load_model('xgboost_model.json')  # Ensure the JSON file is in the same directory

# Load dataset to extract symptoms and diseases
df = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')  # Replace with your dataset's filename

# Extract diseases (first column) and symptoms (remaining columns)
all_diseases = df.iloc[:, 0].unique().tolist()  # Unique disease names from the first column
all_symptoms = df.columns[1:].tolist()         # Symptom names from remaining columns

# Helper function to process user input into a model-compatible format
def prepare_input(selected_symptoms):
    input_vector = np.zeros(len(all_symptoms))  # Initialize a zero vector for symptoms
    for symptom in selected_symptoms:
        if symptom in all_symptoms:
            input_vector[all_symptoms.index(symptom)] = 1  # Set 1 for selected symptoms
    return np.array([input_vector])

# Streamlit app interface
st.set_page_config(layout="wide")
st.sidebar.markdown("# Disease Prediction App")
st.write("# Disease Prediction using XGBoost")

# Symptom selection by the user
selected_symptoms = st.multiselect("Select your symptoms:", options=all_symptoms)

if st.button("Predict"):
    # Prepare the input for the model
    input_vector = prepare_input(selected_symptoms)
    
    # Predict disease using the model
    prediction = model.predict(input_vector)
    predicted_disease = all_diseases[int(prediction[0])]  # Map prediction to disease name

    # Display the result
    st.write(f"### Predicted Disease: {predicted_disease}")
