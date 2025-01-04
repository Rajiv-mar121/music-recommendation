from flask import Flask, request, jsonify, render_template
import streamlit as st
import joblib
import pandas as pd
import os

app = Flask(__name__)

""" sample payload
{
    "Date": [
        "02-02-2022"
    ]
} 
"""

""" # Load the trained model using pickle
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file) """

models_path = 'models'
print(list(os.listdir(f'{models_path}/')))
# Load the trained model
model = joblib.load('models/cross_gradient_booster.joblib')

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
    # print("Rajiv")
    # print(df)
    # df['y'] = df['Date'].apply(lambda x: int(x[-4:]))
    # df['m'] = df['Date'].apply(lambda x: int(x[3:5]))
    # df['d'] = df['Date'].apply(lambda x: int(x[:2]))  
    
    # x = df[['y', 'm','d']]
    # predictions = model.predict(x)
    return jsonify(data.song_name)

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

if __name__ == '__main__':
    app.run(debug=True)
    """ app.run(host='0.0.0.0', port=5000) """