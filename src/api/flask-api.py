from flask import Flask, request, jsonify, render_template
import librosa
import numpy as np
import streamlit as st
import joblib
import pandas as pd
import os
from io import BytesIO
from tensorflow.keras.models import load_model

app = Flask(__name__)


""" # Load the trained model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file) """
    
models_path = 'models/'
model_dictionary = {}
model_dictionary[f"Random Forest"] = models_path+'random_forest.joblib' 
model_dictionary[f"CNN (Standard)"] = models_path+'cnn_model_standard_scaler.joblib' 
#model_dictionary[f"CNN (Custom)"] = models_path+'cnn_model_standard_scaler_custom.h5' 
model_dictionary[f"CNN (Custom)"] = models_path+'cnn_model_standard_scaler_custom.keras'

model_dictionary[f"XG Boost"] = models_path+'cross_gradient_booster.joblib' 
model_dictionary[f"Decission Tree"] = models_path+'decission_trees.joblib' 
model_dictionary[f"KNN"] = models_path+'knn.joblib' 
model_dictionary[f"Logistic Regression"] = models_path+'logistic_regression.joblib' 
model_dictionary[f"Naive Bayes"] = models_path+'naive_bayes.joblib' 
model_dictionary[f"Neural Net"] = models_path+'neural_nets.joblib' 

model_dictionary[f"Stochastic Gradient Descent"] = models_path+'stochastic_gradient_descent.joblib' 
model_dictionary[f"Support Vector Machine"] = models_path+'support_vector_machine.joblib' 

#model_dictionary[f"Decission_trees"] = models_path+'random_forest.joblib' 


#print(list(os.listdir(f'{models_path}/')))
# Load the trained model
#cross_gradient_booster knn
model = joblib.load('models/random_forest.joblib')

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
def recommend_genre():
    print("Inside API")
    #data = request.get_json()
    #audio_path = f'{audio_data_path}/genres_original/reggae/reggae.00036.wav' 
    
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    audio_file = request.files['file']
    model_name = request.form['model_name']
    print("File recieved :"+audio_file.filename)
    print("Model name :"+model_name)
    for key, value in model_dictionary.items():
        if(key == model_name):
            if(model_name == "CNN (Custom)"):
                print("CNN (Custom) found loading", value)
                model = load_model(value)  
                
            else:
                model = joblib.load(value)
            
    user_audio_features = extract_all_features(audio_file)
    #user_audio_features = extract_all_features(audio_path)
    if(model_name == "CNN (Custom)"):
        scaler = joblib.load("models/scaler_cnn_model_standard_scaler_custom.joblib")
        encoder = joblib.load("models/encoder_cnn_model_standard_scaler_custom.joblib")
        #user_audio_features_array = user_audio_features.to_numpy()
        #reshaped_features = user_audio_features_array.reshape(1, -1)
        #user_audio_features = user_audio_features.reshape(1, -1)
        # Standardize features using the saved scaler
        #features_scaled = scaler.transform(reshaped_features)
        
        features_transformed = scaler.transform(np.array(user_audio_features, dtype = float).reshape(1, -1))
        print("UI reshaped: ",features_transformed)
        
        # Below 2 lines are also working but 
        #user_audio_features_np = user_audio_features.to_numpy()
        #features_transformed = user_audio_features_np.reshape(1, 58)
        print(len(user_audio_features))
        feat = [66149,0.3354063630104065,0.09104829281568527,0.1304050236940384,0.0035210042260587215,1773.0650319904662,167541.6308686573,1972.7443881356735,117335.77156332089,3714.560359074519,1080789.8855805045,0.08185096153846154,0.0005576872402394312,-7.848480163374916e-05,0.008353590033948421,-6.816183304181322e-05,0.005535192787647247,129.19921875,-118.62791442871094,2440.28662109375,125.08362579345703,260.9569091796875,-23.443723678588867,364.08172607421875,41.32148361206055,181.69485473632812,-5.976108074188232,152.963134765625,20.115140914916992,75.65229797363281,-16.04541015625,40.22710418701172,17.85519790649414,84.32028198242188,-14.633434295654297,83.4372329711914,10.270526885986328,97.00133514404297,-9.70827865600586,66.66989135742188,10.18387508392334,45.10361099243164,-4.681614398956299,34.169498443603516,8.417439460754395,48.26944351196289,-7.233476638793945,42.77094650268555,-2.8536033630371094,39.6871452331543,-3.2412803173065186,36.488243103027344,0.7222089767456055,38.099151611328125,-5.05033540725708,33.618072509765625,-0.24302679300308228,43.771766662597656]

        #feat = scaler.transform(np.array(feat, dtype = float).reshape(1, -1))
        #print("hardcoded reshaped : ",feat)
        #print(len(feat))
        # predictions = model.predict(np.array(feat, dtype = float).reshape(1, 58))
        # print("Harcoded Predic: ",predictions)
        predictions = model.predict(np.array(features_transformed, dtype = float).reshape(1, 58))
        print("UI predic: ",predictions)
        # predicted_class_index = np.argmax(predictions, axis=1)[0]
        # predicted_genre = encoder.inverse_transform([predicted_class_index])[0]
        predicted_class_index = np.argmax(predictions, axis=1)
        print("CNN predicted_class_index ",predicted_class_index)
        predicted_genre = encoder.inverse_transform(predicted_class_index)
        print("CNN genre ",predicted_genre)
        response_json = {"genre-type": predicted_genre[0]}
        return response_json
        
    else:     
        predictions = model.predict(user_audio_features)
        
        print(predictions) 
        print(" Manupulating")
        print(predictions[0])
        if predictions[0] in range(0, 10):
            print("Classes in the model:", model.classes_)
            label_encoder = joblib.load("models/encoder_xgboost.joblib")
           # predicted_class_index = np.argmax(predictions, axis=1)[0]
            predicted_genre = label_encoder.inverse_transform(predictions)
            print("Xgboost genre "+predicted_genre)
            response_json = {"genre-type": predicted_genre[0]}  
            
        else:
            response_json = {"genre-type": predictions[0]}
            return response_json

        #response_json = {"genre-type": "reggae-hardcoded"}    
        return response_json

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
    #print("calling extarcting" +audio_path.filename)
    #data_column = X.columns

    #y, sr = librosa.load(audio_path)
    # Convert the file to a BytesIO object
    audio_bytes = BytesIO(audio_path.read())
    #, duration=3
    y, sr = librosa.load(audio_bytes, sr=None, duration=3)
    y, _ = librosa.effects.trim(y)
    #duration = librosa.get_duration(y=y, sr=sr)
    #duration = 66149
    #print(np.shape(y))
    #my_duration = 661794/sr
    #print(my_duration)
    duration = len(y)
    
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
    #feature_names = [f"f{i+1}" for i in range(len(features))]
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
    app.run(debug=False)
    #app.run(debug=True, use_reloader=True, extra_files=["./music-recommendation"])
    """ app.run(host='0.0.0.0', port=5000) """