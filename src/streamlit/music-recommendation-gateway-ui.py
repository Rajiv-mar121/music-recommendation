import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
import requests
import re
import matplotlib.pyplot as plt
import plotly.express as px
import json
import random
import librosa
import librosa.display
import io

# Run UI
# streamlit run .\src\streamlit\music-recommendation-gateway-ui.py
# Set page configuration

# Flask API base URL
API_URL = "http://127.0.0.1:5000/api"
image_path = "output/cnn_history.png"
audio_data_path = 'data/audio'

st.set_page_config(
    page_title="Multi-page Streamlit App",
    layout="wide"
)
image_base_dir = 'output/images/'
classification_base_dir = 'output/'
# Create horizontal navigation menu using streamlit-option-menu
def navigation():
    selected = option_menu(
        menu_title=None,
        options=["Home", "Data Visualization","Model Analysis","Model Prediction"],
        icons=["house", "graph-up", "bar-chart-line-fill","graph-up-arrow"],
        default_index=0,
        orientation="horizontal",
        styles={
            "container": {"padding": "0!important", "background-color": "#fafafa"},
            "icon": {"color": "orange", "font-size": "25px"},
            "nav-link": {
                "font-size": "20px",
                "text-align": "center",
                "margin": "0px",
                "--hover-color": "#eee",
            },
            "nav-link-selected": {"background-color": "#0E1117"},
        }
    )
    return selected

def load_image(image_path):
    """Load and return an image if it exists"""
    if os.path.exists(image_path):
        return Image.open(image_path)
    return None

# Home page content
def home():
    st.title("Home")
    
    # Create tabs
    tab1, tab2 = st.tabs(["File Exploration", "Tab 2"])
    
    with tab1:
        st.header("Home Tab 1")
        st.write("Live Audio File Exploration")
        # st.markdown("""
        # Exploring Audio file:
        # - Feature 1
        # - Feature 2
        # - Feature 3
        # """)
        # if st.button("Show Image 1"):       
        #     image_path = "output/cnn_history.png"
        #     image = load_image(image_path)
        #     if image:
        #         st.image(image, caption="Image 1", use_column_width=True)
        #     else:
        #         st.error("Image 1 not found. Please check the file path.")
        uploaded_file = st.file_uploader("Upload a sound file (MP3/WAV):", type=["mp3", "wav"])
        if uploaded_file:
            st.write(f"Uploaded file: {uploaded_file.name}")   
            if st.button("Visualize File"):
                with st.spinner("Processing audio file... Please wait ⏳"):
                    y, sr = librosa.load(uploaded_file, sr=None)

                    # Compute Spectrogram (STFT)
                    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
                    
                    # plt.figure(figsize = (16, 6))
                    # librosa.display.waveshow(uploaded_file, sr=sr, alpha=0.4, color = '#A300F9');
                    # plt.plot(t, normalize(spectral_centroids), color='#FFB100');

                    # Compute Chroma STFT
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr)

                    # Compute RMS Energy
                    rms = librosa.feature.rms(y=y)
                    # with st.spinner('Loading Graphs Please wait...'):
                    #     time.sleep(3)
                    # Create subplots
                    fig, ax = plt.subplots(3, 1, figsize=(6, 6))

                    # Plot Spectrogram
                    img1 = librosa.display.specshow(D, sr=sr, x_axis="time", y_axis="log", ax=ax[0])
                    ax[0].set_title("Spectrogram (STFT)", fontsize=8)
                    fig.colorbar(img1, ax=ax[0], format="%+2.0f dB")

                    # Plot Chroma STFT
                    img2 = librosa.display.specshow(chroma, sr=sr, x_axis="time", y_axis="chroma", ax=ax[1])
                    ax[1].set_title("Chroma STFT", fontsize=8)
                    fig.colorbar(img2, ax=ax[1])

                    # Plot RMS Energy
                    ax[2].plot(librosa.times_like(rms, sr=sr), rms[0], color="r")
                    ax[2].set_title("RMS Energy", fontsize=8)
                    ax[2].set_xlabel("Time (s)", fontsize=8)
                    ax[2].set_ylabel("RMS", fontsize=8)
                    
                    plt.tight_layout() 

                    # Display in Streamlit
                    st.pyplot(fig)
                    
                st.success("Rendering Complete ✅") 
                
        
    with tab2:
        st.header("Home Tab 2")
        st.write("Welcome to the second tab of the Home page!")
        st.markdown("""
        This is sample content for Home Tab 2:
        - Section A
        - Section B
        - Section C
        """)

# Visualization page content
def visualization():
    st.title("Audio Dataset Visualization")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Audio file Exploration", "Features Exploration"])
    with tab1:
        #st.header("Audio file Exploration", divider=False)
        st.markdown(
            "<h1 style='text-align: center;'>Audio file Exploration</h1>",
            unsafe_allow_html=True
        )
        image_paths = [
            image_base_dir+"sound_waves_reggae36.png",
            image_base_dir+"spectrogram.png",
            image_base_dir+"sftf.png",
            image_base_dir+"metal_mel_spectrogram.png",
            image_base_dir+"classical_mel_spectrogram.png",
            image_base_dir+"harmonics_and_perceptrual.png"

        ]
        if st.button("Show Audio Graphs"): 
            with st.spinner('Loading Please wait...'):
                time.sleep(3)
            #st.success("Done!")      
            st.image(image_paths, use_column_width=False,width = 1000)

        
    with tab2:
        #st.header("Features Exploration",divider=False)
        # col1, col2, col3 = st.columns([1, 2, 1])
        st.markdown(
            "<h1 style='text-align: center;'>Features Exploration</h1>",
            unsafe_allow_html=True
        )
        
        feature_image_paths = [       
            image_base_dir+"harmonics_and_perceptrual.png",
            image_base_dir+"spectral_centroid.png",
            image_base_dir+"spectral_rolloff.png",
            image_base_dir+"mfcc.png",
            image_base_dir+"mfcc_scaled.png",
            image_base_dir+"chroma_features.png",
            image_base_dir+"corr_heatmap.png",
            image_base_dir+"BPM_Boxplot.png",
            image_base_dir+"PCA_Scattered.png"

        ]

        if st.button("Show Audio features Graphs"):
            with st.spinner('Loading Please wait...'):
                time.sleep(3)
                #st.progress(10)
            #st.success("Done!")       
            st.image(feature_image_paths, use_column_width=False,width = 1000)
            


# Model Analysis page content
def model_analysis():
    st.title("Model Analysis")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Classification Reports", "Best Models"])
    accuracy_data = []
    with tab1:
        st.header("All Models Performance Metrics")

        if os.path.exists(classification_base_dir):
            text_files = [f for f in os.listdir(classification_base_dir) if f.endswith(".txt")]
                # Show the total number of text files
            total_files = len(text_files)
            st.subheader(f"Total Model found: {total_files}")
            
            if text_files:
                 for text_file in text_files:
                     file_path = os.path.join(classification_base_dir, text_file)
                     # Read the content of each text file
                     with open(file_path, "r", encoding="utf-8") as file:
                        content = file.read()
                     # Extracting accuracy ie :  accuracy                           0.67
                     match = re.search(r'accuracy\s*(\d*\.\d+|\d+)', content)
                     file_name_without_extension = os.path.splitext(text_file)[0]
                     if match:
                         accuracy = float(match.group(1))
                         accuracy_data.append((file_name_without_extension, accuracy))
                        
                     st.subheader(f"Model : {file_name_without_extension}")
                     
                     #st.text_area(f"Content of {text_file}", content, height=300)
                     st.text_area(f"", content, height=300,  key=text_file, help="File content here", label_visibility="collapsed")
                     #st.text(content)

            else:
                st.warning("No text files found in the specified directory.")

        else:
            st.error(f"Directory '{classification_base_dir}' does not exist.")
        # Sample data for visualization
        # chart_data = pd.DataFrame(
        #     np.random.randn(20, 3),
        #     columns=['A', 'B', 'C']
        # )
        # st.line_chart(chart_data)
    
    with tab2:
        st.header("Best Model ") 
        accuracy_data.sort(key=lambda x: x[1], reverse=True)
        # st.subheader("Top 3 Models with accuracy")
        # for idx, (file, accuracy) in enumerate(accuracy_data[:3]):
        #     st.write(f"{idx + 1}. {file} - Accuracy: {accuracy:.2f}") 
            
        df = pd.DataFrame(accuracy_data, columns=["Model", "Accuracy"])
        df = df.sort_values(by="Accuracy", ascending=False)
        
         # Show top 5 files
        st.subheader("Top 5 Most Accurate Models are")
        st.table(df.head(5))

        
        # Plot the graph
        # st.subheader("Accuracy Graph")
        # plt.figure(figsize=(8, 3))
        # plt.bar(df["Model"], df["Accuracy"], color="skyblue")
        # plt.xlabel("Models")
        # plt.ylabel("Accuracy")
        # plt.title("Accuracy of Models")
        # plt.xticks(fontsize = 10)
        # plt.xticks(rotation=60)
        # st.pyplot(plt)
        
        fig = px.bar(
            df,
            x="Model",
            y="Accuracy",
            text="Accuracy",
            color="Accuracy",
            color_continuous_scale="Blues",
            labels={"Accuracy": "Accuracy (%)"},
            title="Accuracy by Models",
        )

        # Add percentage formatting to the tooltip and bar labels
        fig.update_traces(
            texttemplate="%{text:.2%}",  # Show percentages on the bars
            hovertemplate="<b>%{x}</b><br>Accuracy: %{y:.2%}<extra></extra>",
        )
        # Customize layout for clarity and fit
        fig.update_layout(
            xaxis=dict(title="Models", tickangle=-45),
            yaxis=dict(title="Accuracy"),
            font=dict(size=18),
            margin=dict(l=40, r=40, t=40, b=80),
            height=800,
        )
        st.plotly_chart(fig, use_container_width=True)
        
       
        
        # Display the table
        st.subheader("Accuracy Data Table")
        st.dataframe(df)
        #st.dataframe(df.style.set_properties(**{"font-size": "18px"}))
       
# Model Analysis page content
def model_recommendation():
    st.title("Model Recommendation")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Genre Pridiction", "Similar Song"])
    
    with tab1:
        st.header("Music Recommendation and Genre Classification")
        audio_data_path = 'data/audio'
        with st.form("model_form", clear_on_submit=True):
            st.write("Model Selection Form:")
    
            # Dropdown box
            options = ["CNN (Standard)","CNN (Custom)", "LTSM (RNN)","Random Forest","Transformer-wav2vec" ,"XG Boost","Decission Tree", "KNN","Logistic Regression","Naive Bayes","Neural Net","Stochastic Gradient Descent","Support Vector Machine"]
            selected_model = st.selectbox("Choose an model:", options,index=None,
                placeholder="...PlEASE SELECT...",)
            
            # Text input box
            #user_input = st.text_input("Enter song name:")
            
            uploaded_file = st.file_uploader("Upload a sound file (MP3/WAV):", type=["mp3", "wav"])
            # Submit button
            submit_button = st.form_submit_button("Submit")
            
        if uploaded_file:
            st.write(f"Uploaded file: {uploaded_file.name}")   
            if submit_button:
                st.write(f"You selected: {selected_model}")
                #st.write(f"Your input: {user_input}") 
                model_data = {
                    "model_name": selected_model
                }  
                files = {
                    "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
                }
                try:   
                    response = requests.post(f"{API_URL}/recommend", files=files, data=model_data)
                    with st.spinner('Loading Please wait...'):
                        time.sleep(3)
                    if response.status_code == 200:
                        st.success("song recommender initiated")
                        st.json(response.json())
                        #st.write(f"The predicted genre is: **{response}**")
                        #st.text_area(response.content)
                        
                        time.sleep(3)
                        recommend_song(response.json())
                    else:
                        st.error(f"Failed to send data: {response.status_code}")
                except Exception as e:
                    st.error(f"Error: {e}")     
    with tab2:
        st.header("Similar Song Recommendation")
        song_name = st.text_input("Enter the song name")
        if st.button("Submit"):
            try:
                payload = {"song_name": song_name}
                response = requests.post(f"{API_URL}/recommend-song", json=payload)
                if response.status_code == 200:
                    st.success("Similar song recommender initiated")
                    response_json = response.json()
                    #st.json(response_json)
                    sorted_data = dict(sorted(response_json.items(), key=lambda item: item[1], reverse=True))
                    st.json(sorted_data)
                    # Put loop and play songs
                    # Assuming songs are stored locally or on a URL
                    recommended_song = response_json.keys()
                    # Initialize session state for song playback
                    for song in recommended_song:
                        genre = song.split('.')[0]
                        #print(genre)
                        audio_file = os.path.join(audio_data_path, "genres_original", genre, song)
                        #print(audio_file)
                        # Audio Player
                        #st.audio(audio_file, format='audio/wav')
                        with st.container():
                            st.subheader(f"🎶 {song}")  # Display song name
                            st.write(f"**Genre:** {genre.capitalize()}")  
                            st.audio(audio_file, format='audio/wav')  # Audio player
                        st.markdown("---")  # Separator for better UI
                        
                    #audio_file = f'{audio_data_path}/genres_original/pop/pop.00023.wav'
                else:
                    st.error(f"Failed to send data: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
                
def recommend_song(genre_json):
    # data = json.loads(genre_json)
    genre = genre_json["genre-type"]
    print(genre)
    num = random.randint(0, 48)
    formatted_num = f"{num:02}" 
    payload = {"song_name": genre.lower()+".000"+formatted_num+".wav"}
    response = requests.post(f"{API_URL}/recommend-song", json=payload)
    with st.spinner('Calculating Similar Songs Please wait...'):
        time.sleep(3)
    if response.status_code == 200:
        st.success("Similar song recommender initiated")
        response_json = response.json()
        #st.json(response_json)
        sorted_data = dict(sorted(response_json.items(), key=lambda item: item[1], reverse=True))
        st.json(sorted_data)
        # Put loop and play songs
        # Assuming songs are stored locally or on a URL
        recommended_song = response_json.keys()
        # Initialize session state for song playback
        for song in recommended_song:
            genre = song.split('.')[0]
            #print(genre)
            audio_file = os.path.join(audio_data_path, "genres_original", genre, song)
            #print(audio_file)
            # Audio Player
            #st.audio(audio_file, format='audio/wav')
            with st.container():
                st.subheader(f"🎶 {song}")  # Display song name
                st.write(f"**Genre:** {genre.capitalize()}")  
                st.audio(audio_file, format='audio/wav')  # Audio player
            st.markdown("---")  # Separator for better UI
# Main app
def main():
    # Handle navigation
    choice = navigation()
    
    # Render content based on selection
    if choice == "Home":
        home()
    elif choice == "Data Visualization":
        visualization()
    elif choice == "Model Analysis":
        model_analysis()
    elif choice == "Model Prediction":
        model_recommendation()
if __name__ == "__main__":
    main()