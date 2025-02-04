import pandas as pd
import numpy as np
import librosa
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import IPython.display as ipd
from IPython.display import Audio
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRFClassifier
from tensorflow import keras
from keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import joblib
import os
import io
from contextlib import redirect_stdout

##Execute
# python .\src\music-recommender-LTSM.py

audio_data_path = 'data/audio'
output_path = 'output/'
plt_image_path = output_path+'images/'
model_path = "models/"

 #Plotting the curves
def plotValidate(history, filename):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    #plt.show()
    plt.title((os.path.splitext(filename)[0]).capitalize())
    plt.savefig(plt_image_path+filename, dpi=300, bbox_inches="tight")
    plt.close() 

def create_ltsm_model(): 
    data_ltsm = pd.read_csv(f'{audio_data_path}/audio_features_3_sec_extracted.csv')
    # Drop File name
    data_ltsm = data_ltsm.drop(labels='filename',axis=1)
    data_ltsm.head()
    genre_label = data_ltsm.iloc[:, -1] #(Select last col only) 'label')
    label_encoder = LabelEncoder()
    #Fitting the label encoder & return encoded labels
    y_genre_label_encoded = label_encoder.fit_transform(genre_label)
    ## Saving encoder
    joblib.dump(label_encoder, model_path+"encoder_ltsm_model.joblib")
    
    original_labels = label_encoder.inverse_transform(y_genre_label_encoded)
    print("Decoded labels:", original_labels)
    standard_scaler = StandardScaler()
    #features = data_ltsm.drop(columns=['label'])
    X_transform = standard_scaler.fit_transform(np.array(data_ltsm.iloc[:, :-1], dtype = float))
    ## Save standrad scaler
    #joblib.dump(standard_scaler, model_path+"scaler_ltsm_model.joblib")
    print(X_transform.shape)
    
    # Reshape data for LSTM (samples, time steps, features)
    X = X_transform.reshape((X_transform.shape[0], 1, X_transform.shape[1]))
    y = y_genre_label_encoded
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define LSTM model
    ltsm_model = Sequential([
    LSTM(128, return_sequences=True, input_shape=(X.shape[1], X.shape[2])),
    Dropout(0.3),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dense(len(np.unique(y)), activation='softmax')  # Output layer for classification
    ])
    ltsm_summary = ltsm_model.summary()
    with open(output_path+'ltsm_architecture.txt','w') as f:
        with redirect_stdout(f):
            ltsm_model.summary()
   
            
    # Compile the model
    ltsm_model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Define early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor validation loss
        patience=5,          # Number of epochs with no improvement before stopping
        restore_best_weights=True  # Restore the best model weights
    )
    # Train the model
    epochs =50
    batch_size = 32
    ltsm_model_history = ltsm_model.fit(
                            X_train, y_train, 
                            epochs=epochs, 
                            batch_size=batch_size, 
                            validation_data=(X_test, y_test),
                            callbacks=[early_stopping])
    
    #Saving model
    ltsm_model.save(model_path+"ltsm_model.keras")
    
    # Evaluate the model
    test_loss, accuracy = ltsm_model.evaluate(X_test, y_test)
    print("The test loss is :",test_loss)
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    predicted_value = ltsm_model.predict(X_test)
    preds_class = np.argmax(predicted_value, axis=1) 
    target_names = sorted(set(y_genre_label_encoded))
    report = classification_report(y_test, preds_class, labels=target_names)
    print(f'Testing:\n {report}')
    # Save the report to a text file
    with open(output_path+"ltsm_classification_report_custom.txt", "w") as file:
        file.write(report)
    plotValidate(ltsm_model_history, "ltsm_model_history.png")
        
    ## Pending save history images

    
### Main Entry Point #######
# Main code

if __name__ == "__main__":
    create_ltsm_model();