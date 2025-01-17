from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import os

app = Flask(__name__)


""" # Load the trained model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file) """

models_path = 'models'
#print(list(os.listdir(f'{models_path}/')))
# Load the trained model
#cross_gradient_booster knn
model = joblib.load('models/cross_gradient_booster.joblib')

# Load audio 
audio_data_path = 'data/audio'
#print(list(os.listdir(f'{audio_data_path}/')))

# Sample GET endpoint
@app.route('/api/data', methods=['GET'])
def get_data():
    return jsonify({"message": "Hello from Flask!", "data": [1, 2, 3, 4, 5]})


# Sample POST endpoint
@app.route('/api/echo', methods=['POST'])
def echo_data():
    payload = request.json
    return jsonify({"received": payload})

@app.route('/api/recommend', methods=['POST'])
def recommend_song():
    data = request.get_json()
    print(data)
    # df = pd.DataFrame(data)
    audio_path = f'{audio_data_path}/genres_original/reggae/reggae.00036.wav' 
    user_audio_features = extract_all_features(audio_path)
    predictions = model.predict(user_audio_features)
    print(predictions) 
    return data

@app.route('/api/predict', methods=['POST'])
def predict():
    data = request.get_json()
   
    print(data)
    df = pd.DataFrame(data)
    print("Rajiv")
    print(df)
    df['y'] = df['Date'].apply(lambda x: int(x[-4:]))
    df['m'] = df['Date'].apply(lambda x: int(x[3:5]))
    df['d'] = df['Date'].apply(lambda x: int(x[:2]))  
    
    x = df[['y', 'm','d']]
    predictions = model.predict(x)
    return jsonify(predictions.tolist())

def extract_all_features(audio_path):
    print("calling extarcting")
    #data_column = X.columns
    y, sr = librosa.load(audio_path)
    y, _ = librosa.effects.trim(y)
    duration = librosa.get_duration(y=y, sr=sr)
    #print(duration)
    # Chroma
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)

    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)

    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)

    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)

    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)

    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)

    # Harmonic and Percussive
    harmony, perc = librosa.effects.hpss(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    perceptr_mean = np.mean(perc)
    perceptr_var = np.var(perc)

    # Tempo
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
   

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)
    #print(mfcc_mean)
    #print(mfcc_var)

    # Combine all features into a single vector 19 features
    features = np.hstack((
        duration, chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, 
        spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, 
        rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, 
        mfcc_mean, mfcc_var
    ))
    feature_names = [f"f{i+1}" for i in range(len(features))]
    #min_max_scaler = preprocessing.MinMaxScaler()
    columns = [
    'length', 'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
    'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
    'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
    'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
    'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
    'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
    'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
    'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
    'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
    'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
]
    # Create x_test DataFrame
    print("Creating frame")
    user_audio_features = pd.DataFrame([features], columns=columns)
    print("After Creating frame")
    # Do not scale on single row because it will give always 0  (X-Xmin)/Xmax - Xmin 
    #np_scaled = min_max_scaler.fit_transform(x_test) 
    #print(np_scaled)
    
    #x_test = pd.DataFrame(np_scaled, columns=feature_names)
    print(user_audio_features)
    #print(x_test.iloc[0:4])
    return user_audio_features



if __name__ == '__main__':
    app.run(debug=True)
    """ app.run(host='0.0.0.0', port=5000) """