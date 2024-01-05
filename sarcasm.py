# sarcasm.py
from transformers import pipeline
import streamlit as st
import pandas as pd

# Load the pre-trained model for sarcasm detection
classifier_sarcasm = pipeline("text-classification", model="ac0hik/sarcasm_camembertfineTuned_model")

def classify_sarcasm(text):
    # Use the loaded model for sarcasm detection
    result = classifier_sarcasm(text)
    return result[0]['label']

def classify_sarcasm_csv(upload_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(upload_file)

    # Check if the expected columns are present in the DataFrame
    expected_columns = ['Titre', 'sous-titre']
    for col in expected_columns:
        if col not in df.columns:
            st.write(f"Error: The CSV file must contain a column named '{col}'.")
            return None

    # Combine 'Titre' and 'sous-titre' columns into a new 'combined_text' column
    df['combined_text'] = df['Titre'] + " " + df['sous-titre']

    # Perform sarcasm detection on each row
    df['IsSarcastic'] = df['combined_text'].apply(classify_sarcasm)

    # Drop the 'combined_text' column
    df.drop(columns=['combined_text'], inplace=True)

    return df

def page_heading():
    st.title("Sarcasm Analysis Page")

    # Descriptive paragraph
    st.write(
        "This page allows you to analyze sarcasm in text. "
        "The model used classifies text into two categories: sarcastic and non-sarcastic."
    )

    # Option to classify raw text
    st.subheader("Classify Raw Text")
    Titre = st.text_area("Enter title", "")
    sous_titre = st.text_area("Enter sub-title", "")
    text = Titre + " " + sous_titre
    if st.button("Compute Sarcasm"):
        result = classify_sarcasm(text)
        st.write("Is Sarcastic:", result)

    # Option to classify CSV file
    st.subheader("Classify CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Perform sarcasm detection on the CSV file
        df_result = classify_sarcasm_csv(uploaded_file)

        # If the validation check failed, do not proceed
        if df_result is not None:
            # Display the original DataFrame
            st.write("Original DataFrame:")
            st.write(df_result[['Titre', 'sous-titre', 'IsSarcastic']])

            # Download button for the resulting CSV
            csv_result = df_result.to_csv(index=False).encode()
            st.download_button(
                label="Download Result CSV",
                data=csv_result,
                file_name="sarcasm_analysis_result.csv",
                key="download_csv",
            )
