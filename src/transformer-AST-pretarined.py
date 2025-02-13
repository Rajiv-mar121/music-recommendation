import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import ASTModel, ASTConfig
from transformers import ASTForAudioClassification, ASTFeatureExtractor
import joblib
import os
import warnings
import librosa

warnings.filterwarnings('ignore')
from datasets import load_dataset
from transformers import (
    Wav2Vec2Processor, 
    Wav2Vec2ForSequenceClassification, 
    TrainingArguments, 
    Trainer
)

# python .\src\transformer-AST-pretarined.py 
# (Audio Spectrogram Transformer) model

audio_data_path = 'data/audio'
output_path = 'output/'
plt_image_path = output_path+'images/'
model_path = "models/"
# Load the processor and model
MODEL_NAME = "facebook/wav2vec2-base"
processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=10)

def create_ast_transformer_model(): 
    torch.manual_seed(42)
    data_transformer = pd.read_csv(f'{audio_data_path}/audio_features_3_sec_extracted.csv')
    
    # Drop File name
    data_transformer = data_transformer.drop(labels='filename',axis=1)
    data_transformer.head()
    genre_label = data_transformer.iloc[:, -1] #(Select last col only) 'label')
    label_encoder = LabelEncoder()
    
    #Fitting the label encoder & return encoded labels
    y_genre_label_encoded = label_encoder.fit_transform(genre_label)
    
    ## Saving encoder
    #joblib.dump(label_encoder, model_path+"encoder_transformer_model.joblib")
    
    original_labels = label_encoder.inverse_transform(y_genre_label_encoded)
    print("Decoded labels:", original_labels)
    
    standard_scaler = StandardScaler()
    #features = data_ltsm.drop(columns=['label'])
    #X_transform = standard_scaler.fit_transform(np.array(data_transformer.iloc[:, :-1], dtype = float))
    X_transform = standard_scaler.fit_transform(data_transformer.iloc[:, :-1])
    ## Save standrad scaler
    #joblib.dump(standard_scaler, model_path+"scaler_ltsm_model.joblib")
    print(X_transform.shape)
    
    # Load pre-trained AST model
    # feature_extractor = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    # model = ASTForAudioClassification.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
    
# Function to preprocess audio
def preprocess_audio(example):
    audio, sr = librosa.load(example["audio"]["path"], sr=16000)  # Resample to 16kHz
    example["input_values"] = processor(audio, sampling_rate=16000, return_tensors="pt").input_values[0]
    example["label"] = example["genre"]
    return example
    
def play():
    dataset = load_dataset("marsyas/gtzan",trust_remote_code=True)
    print(dataset)
    label_mapping = dataset["train"].features["genre"].names
    print(label_mapping)
    # Apply preprocessing
    dataset = dataset.map(preprocess_audio, remove_columns=["audio"])
    


### Main Entry Point #######
# Main code

if __name__ == "__main__":
    #create_ast_transformer_model(); 
    play()