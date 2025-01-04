import streamlit as st
import requests

# Run code streamlit run streamlit_demo.py
# Be at music-recommendation dir
# streamlit run .\src\streamlit\music-recommendation-ui.py 
# http://localhost:8501/

# Flask API base URL
API_URL = "http://127.0.0.1:5000/api"

st.title("Music Recommendation and Genre Classification")
tab1, tab2, tab3 = st.tabs(["Home", "Data", "Visualization"])
image_path = "output/cnn_history.png"
# Home Tab
with tab1:
    st.header("Home")
    st.write("Welcome to the Home tab.")
    st.image(image_path, caption="Sample Image", use_column_width=True)

# Fetch data using a GET request
if st.button("Fetch Data from Flask API"):
    try:
        response = requests.get(f"{API_URL}/data")
        if response.status_code == 200:
            data = response.json()
            st.success("Data fetched successfully!")
            st.json(data)
        else:
            st.error(f"Failed to fetch data: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")

# Send Song to recommend  using a POST request
st.subheader("Song Recommender")
song_name = st.text_input("Enter the song name")

if st.button("Submit"):
    try:
        payload = {"song_name": song_name}
        st.image(image_path, caption="CNN history", use_column_width=True)
        response = requests.post(f"{API_URL}/recommend", json=payload)
        if response.status_code == 200:
            st.success("song recommender initiated")
            st.json(response.json())
        else:
            st.error(f"Failed to send data: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")
text_file_path = "output/cnn_classification_report.txt"
if st.button("Show Classification report"):
    st.title("Display a .txt File in Streamlit")
    # Read and display the content
    try:
        with open(text_file_path, "r") as file:
            content = file.read()
        st.text(content)
    except FileNotFoundError:
        st.error("File not found! Please check the file path.")
# Send data using a POST request
st.subheader("Send Data to Flask API")
user_input = st.text_input("Enter some text", "Hello Flask!")
if st.button("Send Data"):
    try:
        payload = {"user_input": user_input}
        response = requests.post(f"{API_URL}/echo", json=payload)
        if response.status_code == 200:
            st.success("Data sent successfully!")
            st.json(response.json())
        else:
            st.error(f"Failed to send data: {response.status_code}")
    except Exception as e:
        st.error(f"Error: {e}")