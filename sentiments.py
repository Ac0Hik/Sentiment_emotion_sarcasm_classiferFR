# sentiments.py
from transformers import pipeline
import streamlit as st
import pandas as pd

# Load the pre-trained model
classifier = pipeline("text-classification", model="ac0hik/camembert_model")

def classify_sentiments(text):
    # Use the loaded model for sentiment analysis
    result = classifier(text)
    return result[0]['label']

def classify_sentiments_csv(upload_file):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(upload_file)
    print(df.columns)

    # Check if the expected column is present in the DataFrame
    expected_column = 'text'
    if expected_column not in df.columns:
        st.write(f"Error: The CSV file must contain a column named '{expected_column}'.")
        return None

    # Perform sentiment analysis on each row
    df['Sentiment'] = df[expected_column].apply(classify_sentiments)

    return df

def page_heading():
    st.title("Sentiment Analysis Page")

    # Descriptive paragraph
    st.write(
        "This page allows you to analyze sentiment in text in the French language . "
        "The model used classifies text into different sentiment categories: positive, negative and neutral."
    )
    # Option to classify raw text
    st.subheader("Classify Raw Text")
    text = st.text_area("Enter Text", "")
    if st.button("Compute Sentiment"):
        result = classify_sentiments(text)
        st.write("Sentiment:", result)

    # Option to classify CSV file
    st.subheader("Classify CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    if uploaded_file is not None:
        # Perform sentiment analysis on the CSV file
        df_result = classify_sentiments_csv(uploaded_file)

        # If the validation check failed, do not proceed
        if df_result is not None:
            # Display the original DataFrame
            st.write("Original DataFrame:")
            st.write(df_result[['text','Sentiment']])

            # Download button for the resulting CSV
            csv_result = df_result.to_csv(index=False).encode()
            st.download_button(
                label="Download Result CSV",
                data=csv_result,
                file_name="sentiment_analysis_result.csv",
                key="download_csv",
            )
