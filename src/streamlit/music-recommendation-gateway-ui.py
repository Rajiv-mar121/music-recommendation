import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pandas as pd
import time
import os
import requests

# Run UI
# streamlit run .\src\streamlit\music-recommendation-gateway-ui.py
# Set page configuration

# Flask API base URL
API_URL = "http://127.0.0.1:5000/api"
image_path = "output/cnn_history.png"

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
    tab1, tab2 = st.tabs(["Tab 1", "Tab 2"])
    
    with tab1:
        st.header("Home Tab 1")
        st.write("Welcome to the first tab of the Home page!")
        st.markdown("""
        This is sample content for Home Tab 1:
        - Feature 1
        - Feature 2
        - Feature 3
        """)
        if st.button("Show Image 1"):       
            image_path = "output/cnn_history.png"
            image = load_image(image_path)
            if image:
                st.image(image, caption="Image 1", use_column_width=True)
            else:
                st.error("Image 1 not found. Please check the file path.")
        
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
    
    with tab1:
        st.header("All Models Performance Metrics")
        import numpy as np
        import pandas as pd
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
                     file_name_without_extension = os.path.splitext(text_file)[0]   
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
        
# Model Analysis page content
def model_recommendation():
    st.title("Model Recommendation")
    
    # Create tabs
    tab1, tab2 = st.tabs(["Chart 1", "Chart 2"])
    
    with tab1:
        st.header("Music Recommendation and Genre Classification")

        # Fetch data using a GET request
        song_name = st.text_input("Enter the song name")
        if st.button("Submit"):
            try:
                payload = {"song_name": song_name}
                st.image(image_path, caption="CNN history", use_column_width=True)
                response = requests.post(f"{API_URL}/recommend", json=payload)
                if response.status_code == 200:
                    st.success("song recommender initiated")
                    st.json(response.json())
                    #st.write(f"The predicted genre is: **{response}**")
                    #st.text_area(response.content)
                else:
                    st.error(f"Failed to send data: {response.status_code}")
            except Exception as e:
                st.error(f"Error: {e}")
        
        
        
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