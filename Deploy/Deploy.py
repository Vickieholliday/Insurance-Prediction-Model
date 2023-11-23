import streamlit as st
import pandas as pd
import joblib, pickle

# Load the trained model using pickle
with open('RFC_model.pkl', 'rb') as f:
    clf = pickle.load(f)

st.write("Model loaded successfully.")

if clf is not None:
    # Streamlit UI
    st.title("Insurance Prediction App")
    st.write("Welcome to the Insurance Prediction App")

    # Input form
    st.header("Enter Property Details")
    YearOfObservation = st.number_input("Year of Observation")
    Insured_Period = st.number_input("Insured Period")
    Building_Dimension = st.number_input("Building Dimension")
    Building_Type = st.selectbox("Building Type", [1, 2, 3, 4, 5])
    Age_of_Property = st.number_input("Age_of_Property")
    NumberOfWindows = st.number_input("Number of Windows")
    Settlement_Rural = st.number_input("Settlement_Rural")
    Settlement_Urban = st.number_input("Settlement_Urban")
    Building_Fenced_No = st.number_input("Building_Fenced_No")
    Building_Fenced_Yes = st.number_input(" Building_Fenced_Yes")
    Building_Painted_No = st.number_input("Building_Painted_No")
    Building_Painted_Yes = st.number_input("Building_Painted_Yes")
    Garden_No = st.number_input("Garden_No")
    Garden_Yes = st.number_input("Garden_Yes")
    Geo_Code_Frequency = st.number_input("Geo Code Frequency")

    # Convert categorical inputs to binary using provided feature names and values
    input_data = {
        'YearOfObservation': [YearOfObservation],
        'Insured_Period': [Insured_Period],
        'Residential': [1],  # Assuming Residential is 'Yes'
        'Building_Dimension': [Building_Dimension],
        'Building_Type': [Building_Type],  # Provide the original feature
        'NumberOfWindows': [NumberOfWindows],
        'Settlement_Rural': [1],  # Assuming Settlement is 'Rural'
        'Settlement_Urban': [0],  # Assuming Settlement is 'Rural'
        'Building_Fenced_No': [0],  # Assuming Building_Fenced is 'Yes'
        'Building_Fenced_Yes': [1],  # Assuming Building_Fenced is 'Yes'
        'Building_Painted_No': [1],  # Assuming Building_Painted is 'No'
        'Building_Painted_Yes': [0],  # Assuming Building_Painted is 'No'
        'Garden_No': [1],  # Assuming Garden is 'No'
        'Garden_Yes': [0],  # Assuming Garden is 'No'
        'Geo_Code_Frequency': [Geo_Code_Frequency],
        'Age_of_Property': [Age_of_Property]
    }

    # Create the sample DataFrame
    sample_df = pd.DataFrame(input_data)

# Make predictions when button is clicked
    if st.button('Predict'):
            prediction = clf.predict(sample_df)[0]
            prediction_label = "Insured" if prediction == 1 else "Not Insured"

            # Display predictions
            st.header("Prediction")
            st.write(f"The property is predicted to be: {prediction_label}")
else:
    st.write("No valid model available, cannot make predictions.")


