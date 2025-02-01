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
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import joblib
import os
import io
from contextlib import redirect_stdout

##Execute
# python .\src\music-recommender-system.py

audio_data_path = 'data/audio'
output_path = 'output/'
plt_image_path = output_path+'images/'
# Load Data
data = None
X_train = None, 
X_test = None, 
y_train = None, 
y_test = None
similarity_df_names = None
cols = None
def load_data():
    print("Loading data")
    global data 
    
    data = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    # data1 = pd.read_csv(f'{audio_data_path}/audio_features_with_genres.csv')
    # data = pd.concat([data, data1], ignore_index=True)
    data = data.iloc[0:, 1:]  # dropping first column 'filename'
    # data_cnn = data_cnn.drop(labels='filename',axis=1)  # another way

""" Create fearures and split the data into tarin and test """
def ceate_features_target():
    y = data['label'] # genre variable pulling out label column values only as Series type.
    
    #select all columns but not the labels as panda DataFrame with coloum name
    X = data.loc[:, data.columns != 'label'] 
    #### NORMALIZE X ####
    global cols
    # Normalize so everything is on the same scale. 
    cols = X.columns #fetch all column name from Dataframe
    
    min_max_scaler = preprocessing.MinMaxScaler()
    #scale the dataset X to a specified range, typically [0, 1] return  NumPy array
    np_scaled = min_max_scaler.fit_transform(X) 

    # new data frame with the new scaled data and assigns column names from the list cols
    X = pd.DataFrame(np_scaled, columns = cols)
    global  X_train, X_test, y_train, y_test
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train))
    print("X_train_cnn.shape[1] = ", X_train.shape[1])
    print("X_train_cnn.shape = ", X_train.shape)
    ## output 
    ## X_train_cnn.shape[1] =  58
    ## X_train_cnn.shape =  (6993, 58)
    print(len(y_train))
    print(type(X))
    
    first_value = X.iloc[30,1] # or for full row X.iloc[30]
    print(first_value)
    #print(y)

def extract_all_features(audio_path):
    y, sr = librosa.load(audio_path)
   # y = librosa.effects.trim(y)

    duration = librosa.get_duration(y=y, sr=sr)
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
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_var = np.var(mfcc, axis=1)

    # Combine all features into a single vector
    features = np.hstack((
        duration,chroma_stft_mean, chroma_stft_var, rms_mean, rms_var, 
        spectral_centroid_mean, spectral_centroid_var, spectral_bandwidth_mean, spectral_bandwidth_var, 
        rolloff_mean, rolloff_var, zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var, perceptr_mean, perceptr_var, tempo, 
        mfcc_mean, mfcc_var
    ))
    feature_names = [f"f{i+1}" for i in range(len(features))]
    # Create x_test DataFrame
    user_audio_feature = pd.DataFrame([features], columns=cols)
    print(user_audio_feature)
    return user_audio_feature

""" Prepare models and save it as joblib file"""
def fit_and_save_models(model, title = "Default"):
    model.fit(X_train, y_train)

    # Save models after fitting
    preds = model.predict(X_test)
    #print(confusion_matrix(y_test, preds))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
    report = classification_report(y_test, preds)
    print(f'Classification Report Testing:\n {report}')
    
    # Saving Classification report to output directory
    report_filename = title.capitalize()+"_report.txt"
    with open(output_path+report_filename, "w") as file:
        file.write(report)
    file_name = title.lower()+'.joblib'
    joblib.dump(model, "models/"+file_name)


def fit_and_save_models_xgboost(model, X_train, y_train, y_test ,title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
    
    # Saving Classification report to output directory
    report = classification_report(y_test, preds)
    print(f'Classification Report Testing:\n {report}')
    report_filename = title.capitalize()+"_report.txt"
    with open(output_path+report_filename, "w") as file:
        file.write(report)

    #Save model
    file_name = title.lower()+'.joblib'
    joblib.dump(model, "models/"+file_name)


def create_models():
    
    model_dictionary = {}

    # Decission trees
    decision_tree = DecisionTreeClassifier()

    # Random Forest
    randomforest = RandomForestClassifier(n_estimators=1000, max_depth=10, random_state=0)

    # Naive Bayes
    naive_bayes = GaussianNB()

    # Stochastic Gradient Descent
    sgd = SGDClassifier(max_iter=5000, random_state=0)
    # KNN
    knn = KNeighborsClassifier(n_neighbors=19)
    # Support Vector Machine
    svm = SVC(decision_function_shape="ovo")
    # Logistic Regression
    logistic_reg = LogisticRegression(random_state=0, solver='lbfgs', multi_class='multinomial')
    
    # Neural Nets An MLPClassifier is created with two hidden layer of 5000 and 10 neurons
    # lbfgs Limited-memory Broyden–Fletcher–Goldfarb–Shanno
    """ Alternatives to 'lbfgs' include 'adam' (default) and 'sgd', 
    which are also optimization methods, but 'lbfgs' is typically used 
    for smaller datasets or when a more accurate result is needed 
    at the cost of computational complexity."""
    # alpha L2 regularization term.  1e-5 (which is 0.00001 )
    # random seed used for initializing the random number
    
    neural_net_mlcp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5000, 10), random_state=1)
    # Putting all models in dictionary 
    model_dictionary[f"Decission_trees"] = decision_tree 
    model_dictionary[f"Random_Forest"] = randomforest 
    # model_dictionary[f"Naive_Bayes"] = naive_bayes 
    # model_dictionary[f"Stochastic_Gradient_Descent"] = knn
    # model_dictionary[f"KNN"] = sgd 
    # model_dictionary[f"Support_Vector_Machine"] = svm
    # model_dictionary[f"Logistic_Regression"] = logistic_reg
    # model_dictionary[f"Neural_Nets"] = neural_net_mlcp 

    # Iterate over the dictionary
    for key, value in model_dictionary.items():
        print(f"Key: {key}, Value: {value}")
        fit_and_save_models(value, key)

    
def create_xgboostmodels():
    # Final model
    label_encoder = LabelEncoder()
    # Fit and transform the training labels
    y_train_encoded = label_encoder.fit_transform(y_train)
    print(y_train_encoded)

    # Transform the test labels (if needed)
    #y_test_encoded = label_encoder.transform(y_test) 
    y_test_encoded = label_encoder.fit_transform(y_test) 
    ## Saving encoder
    joblib.dump(label_encoder, "models/encoder_xgboost.joblib")
    # Cross Gradient Booster
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    fit_and_save_models_xgboost(xgb, X_train, y_train_encoded,y_test_encoded,"Cross_Gradient_Booster")

    # Cross Gradient Booster (Random Forest)
    xgbrf = XGBRFClassifier(objective= 'multi:softmax')
    fit_and_save_models_xgboost(xgbrf, X_train, y_train_encoded,y_test_encoded,"Cross_Gradient_Booster_Random_Forest")

    #print(y_train_encoded)    


 #Plotting the curves
def plotValidate(history, filename):
    print("Validation Accuracy",max(history.history["val_accuracy"]))
    pd.DataFrame(history.history).plot(figsize=(12,6))
    #plt.show()
    plt.title((os.path.splitext(filename)[0]).capitalize())
    plt.savefig(plt_image_path+filename, dpi=300, bbox_inches="tight")
    plt.close()   

#The loss is calculated using sparse_categorical_crossentropy function
def trainNeuralModel(model, X_train_cnn, y_train_cnn, X_test_cnn , y_test_cnn, epochs, optimizer):
    batch_size = 128
    model.compile(optimizer=optimizer,
                  loss='sparse_categorical_crossentropy',
                   metrics=['accuracy']
    )
    return model.fit(X_train_cnn, y_train_cnn, validation_data=(X_test_cnn, y_test_cnn), epochs=epochs, batch_size=batch_size)


def create_cnn_model():

    label_encoder = LabelEncoder()
    y_train_cnn = label_encoder.fit_transform(y_train)
    print(y_train_cnn)

    y_test_cnn = label_encoder.fit_transform(y_test)
    print(X_train.head())
    print(y_test_cnn)

    X_train_cnn = X_train.astype('float32') / 255.0
    X_test_cnn = X_test.astype('float32') / 255.0

    cnn_model = keras.models.Sequential([
    keras.layers.Input(shape=(X_train_cnn.shape[1],)),
    keras.layers.Dense(512, activation="relu"),    
    #keras.layers.Dense(512, activation="relu", input_shape=(X_train_cnn.shape[1],)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(10, activation="softmax"),
    
    ])
    
    print(cnn_model.summary())
    model_history = trainNeuralModel(cnn_model, X_train_cnn, y_train_cnn, X_test_cnn , y_test_cnn, epochs=1500, optimizer='adam')
    print(model_history)
    joblib.dump(cnn_model, "models/cnn_model.joblib")
    preds = cnn_model.predict(X_test_cnn)
    #print("CNN Predicts :")
    #print(preds)
    test_loss, test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn, batch_size=128)
    print("The test loss is :",test_loss)
    print("\nThe test Accuracy is :",test_accuracy*100)
    set_test = set(y_test_cnn)
    set_train = set(y_train_cnn)
    merged_set = set_test.union(set_train)
    target_names = sorted(set(merged_set))
    preds_class = np.argmax(preds, axis=1)  # converting back to unique class
    print(preds_class)
    report = classification_report(y_test_cnn, preds_class, labels=target_names)
    print(f'Testing:\n {report}')
    # Save the report to a text file
    with open(output_path+"cnn_classification_report.txt", "w") as file:
        file.write(report)
    plotValidate(model_history, "cnn_history.png")

def create_cnn_model_with_standardscaler():
    data_cnn = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    data_cnn.head()
    data_cnn = data_cnn.drop(labels='filename',axis=1)
    data_cnn.head()
    class_list = data_cnn.iloc[:, -1] #(Select last col only) 'label')
    convertor = LabelEncoder()
    #Fitting the label encoder & return encoded labels
    y_transform = convertor.fit_transform(class_list)
    #Standard scaler is used to standardize features & look like standard normally distributed data
    fit = StandardScaler()
    X_transform = fit.fit_transform(np.array(data_cnn.iloc[:, :-1], dtype = float))
    
    # Now Split
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_transform, y_transform, test_size=0.33)
    cnn_model = keras.models.Sequential([
    #keras.layers.Input(shape=(X_train_cnn.shape[1],)),
    #keras.layers.Dense(512, activation="relu"),     
    keras.layers.Dense(512, activation="relu", input_shape=(X_train_cnn.shape[1],)),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(256,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(128,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(64,activation="relu"),
    keras.layers.Dropout(0.2),
    
    keras.layers.Dense(10, activation="softmax"),
    # 512 , 256,128, 64,10 are neurons
    # 1 input layer, 3 hidden Layer , 1 output layer
    # Dropout layers after each fully connected layer to prevent overfitting.
    # The Rectified Linear Unit (ReLU) non-linearity and helps the model learn complex patterns. 
    # It outputs max(0, x) for each input.
    ])
    print(cnn_model.summary())

    model_history_standard_scaler = trainNeuralModel(cnn_model, X_train_cnn, y_train_cnn, X_test_cnn , y_test_cnn, epochs=1500, optimizer='adam')
    print(model_history_standard_scaler)
    joblib.dump(cnn_model, "models/cnn_model_standard_scaler.joblib")
    preds = cnn_model.predict(X_test_cnn)
    print("CNN Predicts :")
    print(preds)
    test_loss, test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn, batch_size=128)
    print("The test loss is :",test_loss)
    print("\nThe test Accuracy is :",test_accuracy*100)
    target_names = sorted(set(y_transform))
    preds_class = np.argmax(preds, axis=1)  # converting back to unique class
    report = classification_report(y_test_cnn, preds_class, labels=target_names)
    print(f'Testing:\n {report}')
    # Save the report to a text file
    with open(output_path+"cnn_classification_report_scaler.txt", "w") as file:
        file.write(report)
    plotValidate(model_history_standard_scaler, "cnn_history_standrad_scaler.png")
    
def create_cnn_model_with_custom(): 
    data_cnn = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    data1_cnn = pd.read_csv(f'{audio_data_path}/audio_features_with_genres.csv')
    data_cnn = pd.concat([data_cnn, data1_cnn], ignore_index=True)
    print(data_cnn.shape)
    #data_cnn.head()
    data_cnn = data_cnn.drop(labels='filename',axis=1)
    #data_cnn = data_cnn.drop(labels='length',axis=1)
    data_cnn.head()
    class_list = data_cnn.iloc[:, -1] #(Select last col only) 'label')
    convertor = LabelEncoder()
    #Fitting the label encoder & return encoded labels
    y_transform = convertor.fit_transform(class_list)
    #Standard scaler is used to standardize features & look like standard normally distributed data
    
    ## Saving encoder
    joblib.dump(convertor, "models/encoder_cnn_model_standard_scaler_custom.joblib")
    original_labels = convertor.inverse_transform(y_transform)
    print("Decoded labels:", original_labels)
    standard_scaler = StandardScaler()
    X_transform = standard_scaler.fit_transform(np.array(data_cnn.iloc[:, :-1], dtype = float))
    ## Save standrad scaler
    joblib.dump(standard_scaler, "models/scaler_cnn_model_standard_scaler_custom.joblib")
    # Now Split
    X_train_cnn, X_test_cnn, y_train_cnn, y_test_cnn = train_test_split(X_transform, y_transform, test_size=0.33)
    cnn_model = keras.models.Sequential([
        
    # Added one more layer     and Batch normalization
    # Experiment with batch size  32, 64, or 128
    # keras.layers.Dense(512, activation="relu", input_shape=(X_train_cnn.shape[1],)),
    #X_train_cnn.shape[1] corresponds to num_features like cloumn here it is 58 columns
    keras.layers.Dense(1024, activation="relu", input_shape=(X_train_cnn.shape[1],)),
    #normalization after each dense layer to stabilize and accelerate training
    #normalize the inputs to a layer, stabilizing the learning process and improving convergence speed.
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3), # networks to prevent overfitting. 30% neuron are dropped

    #Add an L2 penalty to the weights to reduce overfitting: , kernel_regularizer=keras.regularizers.l2(0.001)
    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3), 

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation="softmax"),
    
    
    ])
    cnn_summary = cnn_model.summary() 
    #print(cnn_summary)
    # with open(output_path+'cnn_custom_architecture.txt','a') as f:
    #     print(cnn_summary, file=f)
    with open(output_path+'cnn_custom_architecture.txt','w') as f:
        with redirect_stdout(f):
            cnn_model.summary()
    #plot_model(cnn_model, to_file=plt_image_path+"cnn_custom_architecture.png", show_shapes=True, show_layer_names=True)
    optimizer = keras.optimizers.AdamW(learning_rate=0.001)
    model_history_standard_scaler = trainNeuralModel(cnn_model, X_train_cnn, y_train_cnn, X_test_cnn , y_test_cnn, epochs=1500, optimizer=optimizer)
    print(model_history_standard_scaler)
    #joblib.dump(cnn_model, "models/cnn_model_standard_scaler_custom.joblib")
    #cnn_model.save("models/cnn_model_standard_scaler_custom.h5")
    cnn_model.save("models/cnn_model_standard_scaler_custom.keras")
    preds = cnn_model.predict(X_test_cnn)
    print("CNN Custom Predicts :", preds)
    test_loss, test_accuracy = cnn_model.evaluate(X_test_cnn, y_test_cnn, batch_size=128)
    print("The test loss is :",test_loss)
    print("\nThe test Accuracy is :",test_accuracy*100)
    target_names = sorted(set(y_transform))
    preds_class = np.argmax(preds, axis=1)  # converting back to unique class
    report = classification_report(y_test_cnn, preds_class, labels=target_names)
    print(f'Testing:\n {report}')
    # Save the report to a text file
    with open(output_path+"cnn_classification_report_custom.txt", "w") as file:
        file.write(report)
    plotValidate(model_history_standard_scaler, "cnn_history_standrad_scaler_custom.png") 
    
def song_recommender_data():
    global similarity_df_names 
    # Read data
    song_30sec_data = pd.read_csv(f'{audio_data_path}/features_30_sec.csv', index_col='filename')
    # Extract labels
    labels = song_30sec_data[['label']]

    # Drop labels from original dataframe
    song_30sec_data = song_30sec_data.drop(columns=['length','label'])
    print(song_30sec_data.head())
    # Scale the data to make mean = 0 and Std deviation = 1
    data_scaled=preprocessing.scale(song_30sec_data)
    print(data_scaled)

    # Cosine similarity
    similarity = cosine_similarity(data_scaled)
    print("Similarity shape:", similarity.shape)
    # Convert into a dataframe and then set the row index and column names as labels
    sim_df_labels = pd.DataFrame(similarity)
    sim_df_names = sim_df_labels.set_index(labels.index)
    sim_df_names.columns = labels.index
    similarity_df_names = sim_df_names
    print(sim_df_names.head())

def find_similar_songs(song_file_name):
    # Find songs most similar to another song
    series = similarity_df_names[song_file_name].sort_values(ascending = False)
    # Remove cosine similarity == 1 (songs will always have the best match with themselves)
    series = series.drop(song_file_name)
    
    # Display the 5 top matches 
    print("\n*******\nSimilar songs to ", song_file_name)
    print(series.head(5))

def print_label(): 
    data_cnn = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    #data_cnn.head()
    data_cnn = data_cnn.drop(labels='filename',axis=1)
    data_cnn.head()
    class_list = data_cnn.iloc[:, -1] #(Select last col only) 'label')
    convertor = LabelEncoder()
    #Fitting the label encoder & return encoded labels
    y_transform = convertor.fit_transform(class_list)
    #Standard scaler is used to standardize features & look like standard normally distributed data
    original_labels = convertor.inverse_transform(y_transform)
    print("Decoded labels:", original_labels)
    
def get_cnn_model():
    data_cnn = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    print(data_cnn.shape)
    data1_cnn = pd.read_csv(f'{audio_data_path}/audio_features_with_genres.csv')
    data_cnn = pd.concat([data_cnn, data1_cnn], ignore_index=True)
    print(data_cnn.head())
    
    print(data_cnn.shape)
    cnn_model = keras.models.Sequential([
        
    # Added one more layer     and Batch normalization
    # Experiment with batch size  32, 64, or 128
    # keras.layers.Dense(512, activation="relu", input_shape=(X_train_cnn.shape[1],)),
    #X_train_cnn.shape[1] corresponds to num_features like cloumn here it is 58 columns
    keras.layers.Dense(1024, activation="relu", input_shape=(58,)),
    #normalization after each dense layer to stabilize and accelerate training
    #normalize the inputs to a layer, stabilizing the learning process and improving convergence speed.
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3), # networks to prevent overfitting. 30% neuron are dropped

    #Add an L2 penalty to the weights to reduce overfitting: , kernel_regularizer=keras.regularizers.l2(0.001)
    keras.layers.Dense(512, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3), 

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.2),

    keras.layers.Dense(10, activation="softmax"), 
    ])
    cnn_summary = cnn_model.summary() 
    #print(cnn_summary)
    # with open(output_path+'cnn_custom_architecture.txt','w') as f:
    #     with redirect_stdout(f):
    #         cnn_model.summary()
    
# print(list(os.listdir(f'{audio_data_path}/genres_original/')))


### Main Entry Point #######
# Main code

if __name__ == "__main__":
    print("Genre Classification initiated")
    load_data()
    print(data.head())
    ceate_features_target()
    #create_models()
    create_xgboostmodels()
    #create_cnn_model()
    #create_cnn_model_with_standardscaler()
    create_cnn_model_with_custom()
    #print_label()
    #get_cnn_model()
    song_recommender_data()
    find_similar_songs('pop.00019.wav')
    #ipd.Audio(f'{audio_data_path}/genres_original/pop/pop.00019.wav') 
    # Specify the path to the audio file
    audio_file = f'{audio_data_path}/genres_original/pop/pop.00023.wav'

    # Play the audio
    #Audio(audio_file)
    #extract_all_features(f'{audio_data_path}/genres_original/pop/pop.00023.wav')
   
