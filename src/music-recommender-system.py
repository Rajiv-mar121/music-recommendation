import pandas as pd
import numpy as np
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
import IPython.display as ipd
from IPython.display import Audio
from sklearn.metrics.pairwise import cosine_similarity
from xgboost import XGBClassifier, XGBRFClassifier
from tensorflow import keras
from keras.models import Sequential
import matplotlib.pyplot as plt
import joblib
import os


audio_data_path = 'data/audio'
output_path = 'output/'
# Load Data
data = None
X_train = None, 
X_test = None, 
y_train = None, 
y_test = None
similarity_df_names = None
def load_data():
    print("Loading data")
    global data 
    
    data = pd.read_csv(f'{audio_data_path}/features_3_sec.csv')
    data = data.iloc[0:, 1:]  # dropping first column 'filename'
    # data_cnn = data_cnn.drop(labels='filename',axis=1)  # another way

""" Create fearures and split the data into tarin and test """
def ceate_features_target():
    y = data['label'] # genre variable pulling out label column only.
    X = data.loc[:, data.columns != 'label'] #select all columns but not the labels
    #### NORMALIZE X ####

    # Normalize so everything is on the same scale. 
    cols = X.columns
    min_max_scaler = preprocessing.MinMaxScaler()
    np_scaled = min_max_scaler.fit_transform(X)

    # new data frame with the new scaled data. 
    X = pd.DataFrame(np_scaled, columns = cols)
    global  X_train, X_test, y_train, y_test
    # Split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(len(X_train))
    print(len(y_train))
    #print(y)


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
    model_dictionary[f"Naive_Bayes"] = naive_bayes 
    model_dictionary[f"Stochastic_Gradient_Descent"] = knn
    model_dictionary[f"KNN"] = sgd 
    model_dictionary[f"Support_Vector_Machine"] = svm
    model_dictionary[f"Logistic_Regression"] = logistic_reg
    model_dictionary[f"Neural_Nets"] = neural_net_mlcp 

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
    y_test_encoded = label_encoder.transform(y_test) 

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
    plt.savefig("output/"+filename)
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
    model_history = trainNeuralModel(cnn_model, X_train_cnn, y_train_cnn, X_test_cnn , y_test_cnn, epochs=20, optimizer='adam')
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

# print(list(os.listdir(f'{audio_data_path}/genres_original/')))


### Main Entry Point #######
# Main code

if __name__ == "__main__":
    print("Genre Classification initiated")
    load_data()
    print(data.head())
    ceate_features_target()
    create_models()
    create_xgboostmodels()
    create_cnn_model()
    song_recommender_data()
    find_similar_songs('pop.00019.wav')
    ipd.Audio(f'{audio_data_path}/genres_original/pop/pop.00019.wav') 
    # Specify the path to the audio file
    audio_file = f'{audio_data_path}/genres_original/pop/pop.00023.wav'

    # Play the audio
    Audio(audio_file)
   
