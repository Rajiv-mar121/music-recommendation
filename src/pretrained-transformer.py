from flask import Flask, request, jsonify
import torch
import librosa

import numpy as np
from safetensors.torch import load_model
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from transformers import AutoConfig
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

#python .\src\pretrained-transformer.py

audio_data_path = 'data/audio'
models_path = 'models/'
# Load the fine-tuned model and processor
MODEL_PATH = "models/fine-tuned-wav2vec2-genre"

# Load processor from the base model
# MODEL_NAME = "facebook/wav2vec2-base"
# processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)

# # Save the processor from base model in the fine-tuned model directory
# processor.save_pretrained(models_path+"fine-tuned-wav2vec2-genre-6mar")

# Load model
#model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH, from_safetensors=True)

model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH,  num_labels=10)
processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)

# Load Label from models
config = AutoConfig.from_pretrained(MODEL_PATH)
#print(config.id2label)
LABELS = {
    0: "Blues",
    1: "Classical",
    2: "Country",
    3: "Disco",
    4: "HipHop",
    5: "Jazz",
    6: "Metal",
    7: "Pop",
    8: "Reggae",
    9: "Rock"
}

def predict_genre(audio_file):
    # Load audio file
    #audio, sr = librosa.load(audio_file, sr=16000)
   
    # Get total duration of the audio file
    total_duration = librosa.get_duration(filename=audio_file, sr=16000)
    print("Total duration ", total_duration)
    # Load only up to 30 seconds if duration is greater than 30 seconds
    if total_duration > 35:
        audio, sr = librosa.load(audio_file, sr=16000, duration=30)
    else:
        audio, sr = librosa.load(audio_file, sr=16000)
    input_values = processor(audio, return_tensors="pt", sampling_rate=16000).input_values
    
    # Get model predictions
    with torch.no_grad():
        logits = model(input_values).logits
    
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_genre = LABELS.get(predicted_class, "Unknown")
    print("predicted_class = ", predicted_class)
    #print("predicted_genre = ", predicted_genre)
    return predicted_genre  # Update this with a mapping if needed





### Main Entry Point #######
# Main code

if __name__ == "__main__":
    print("Rajiv")
    #audio_file = f'{audio_data_path}/genres_original/pop/pop.00023.wav'
    #audio_file = f'{audio_data_path}/genres_original/blues/blues.00005.wav'
    audio_file = f'{audio_data_path}/genres_original/metal/metal.00056.wav'
    #audio_file = f'data/US_Army_Blues_-_08_-_Kellis_Number(chosic.com).wav'
    predicted_genre = predict_genre(audio_file)
    print("predicted_genre = ", predicted_genre)