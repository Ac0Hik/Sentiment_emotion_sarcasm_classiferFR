# app.py
import streamlit as st

def main():
    st.title("Text Classification App")

    # Select menu for choosing between sentiment, emotions, and sarcasm
    option = st.sidebar.selectbox("Select Option", ["Sentiment Analysis", "Emotion Analysis", "Sarcasm Analysis"])

    if option == "Sentiment Analysis":
        redirect_page("sentiments")
    elif option == "Emotion Analysis":
        redirect_page("emotions")
    elif option == "Sarcasm Analysis":
        redirect_page("sarcasm")

def redirect_page(task):
    try:
        module = __import__(task)
        page_heading = getattr(module, "page_heading")
        page_heading()
    except Exception as e:
        st.write(f"Error: {e}")

if __name__ == "__main__":
    main()
