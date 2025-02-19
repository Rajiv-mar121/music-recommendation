import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import  classification_report
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, GlobalAveragePooling1D
from tensorflow.keras.layers import MultiHeadAttention, Add
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from contextlib import redirect_stdout
import matplotlib.pyplot as plt

##Execute
# python .\src\music-recommender-Transformer.py

audio_data_path = 'data/audio'
output_path = 'output/'
plt_image_path = output_path+'images/'
model_path = "models/"


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.3):
    x = MultiHeadAttention(num_heads=num_heads, key_dim=head_size)(inputs, inputs)
    x = Dropout(dropout)(x)
    x = Add()([x, inputs])  # Skip connection
    x = LayerNormalization()(x)
    
    # Feedforward Network
    x_ff = Dense(ff_dim, activation="relu")(x)
    x_ff = Dropout(dropout)(x_ff)
    x_ff = Dense(inputs.shape[-1])(x_ff)
    x = Add()([x, x_ff])  # Skip connection
    x = LayerNormalization()(x)
    
    return x

# 4. Build the Transformer Model
def build_transformer_model(input_shape, num_classes, num_heads=4, ff_dim=64):
    inputs = Input(shape=input_shape)

    # Transformer Encoder Blocks
    x = transformer_encoder(inputs, head_size=64, num_heads=num_heads, ff_dim=ff_dim)
    x = transformer_encoder(x, head_size=64, num_heads=num_heads, ff_dim=ff_dim)

    # Global Pooling Layer
    x = GlobalAveragePooling1D()(x)

    # Fully Connected Layers
    x = Dense(32, activation="relu")(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    return model

 #Plotting the curves
def plotValidate(history, filename):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    #plt.show()
    plt.title((os.path.splitext(filename)[0]).capitalize())
    plt.savefig(plt_image_path+filename, dpi=300, bbox_inches="tight")
    plt.close() 

def create_transformer_model(): 
    data_transformer = pd.read_csv(f'{audio_data_path}/audio_features_3_sec_extracted.csv')
    # Drop File name
    data_transformer = data_transformer.drop(labels='filename',axis=1)
    data_transformer.head()
    genre_label = data_transformer.iloc[:, -1] #(Select last col only) 'label')
    label_encoder = LabelEncoder()
    #Fitting the label encoder & return encoded labels
    y_genre_label_encoded = label_encoder.fit_transform(genre_label)
    
    ## Saving encoder
    joblib.dump(label_encoder, model_path+"encoder_transformer_model.joblib")
    
    original_labels = label_encoder.inverse_transform(y_genre_label_encoded)
    print("Decoded labels:", original_labels)
    
    standard_scaler = StandardScaler()
    
    #features = data_ltsm.drop(columns=['label'])
    #X_transform = standard_scaler.fit_transform(np.array(data_transformer.iloc[:, :-1], dtype = float))
    X_transform = standard_scaler.fit_transform(data_transformer.iloc[:, :-1])
    
    ## Save standrad scaler
    #joblib.dump(standard_scaler, model_path+"scaler_ltsm_model.joblib")
    print(X_transform.shape)
    
    # Reshape input to (samples, time_steps, features)
    # Here we assume a fixed time step of 10 (change as needed)
    #  X.shape[0] = all rows around 7k  X.shape[1] = 58 columns
    num_features = X_transform.shape[1]
    #time_steps = 29
    #time_steps = 2
    time_steps = 1
    feature_dim = num_features // time_steps
    if num_features % time_steps != 0:
        raise ValueError(f"Feature size ({num_features}) is not divisible by time_steps ({time_steps}). "
                     f"Try a different time_steps value.")
    
    
    X = X_transform.reshape(( -1, time_steps, feature_dim))
    y = y_genre_label_encoded
    print("After reshape ",X_transform.shape)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    num_classes = len(np.unique(y))
    print("Start Building ")
    
    # Initialize and compile the model
    transformer_model = build_transformer_model(input_shape=(X.shape[1], X.shape[2]), num_classes=num_classes)
    transformer_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    
    # Print model summary
    transformer_model.summary()
    with open(output_path+'transformer_architecture.txt','w') as f:
        with redirect_stdout(f):
            transformer_model.summary()
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,          # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore the best model weights
    )
    #,callbacks=[early_stopping]
    # 5. Train the Model
    transformer_model_history = transformer_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=110, batch_size=32)

    # 6. Evaluate the Model
    test_loss, test_acc = transformer_model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f'Test Accuracy: {test_acc * 100:.2f}%')      # Test Accuracy: 80.22%
    predicted_value = transformer_model.predict(X_test)
    preds_class = np.argmax(predicted_value, axis=1) 
    target_names = sorted(set(y_genre_label_encoded))
    report = classification_report(y_test, preds_class, labels=target_names)
    print(f'Testing:\n {report}')
    with open(output_path+"transformer_classification_report_custom.txt", "w") as file:
        file.write(report)
    plotValidate(transformer_model_history, "transformer_model_history.png")

    
  ### Main Entry Point #######
# Main code

if __name__ == "__main__":
    create_transformer_model();  
