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
from xgboost import XGBClassifier, XGBRFClassifier
import joblib
import os


audio_data_path = 'data/audio'
# Load Data
data = None
X_train = None, 
X_test = None, 
y_train = None, 
y_test = None
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
    file_name = title+'.joblib'
    joblib.dump(model, "models/"+file_name)


def fit_and_save_models_xgboost(model, X_train, y_train, y_test ,title = "Default"):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    #y_pred_test = model.predict(X_test)
    #target_names = sorted(set(y))
    print('Accuracy', title, ':', round(accuracy_score(y_test, preds), 5), '\n')
    #print(f'Testing:\n {classification_report(y_test, y_pred_test, labels=target_names)}')
    file_name = title+'.joblib'

    #Save model
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
   
    
    

    #model_dictionary[f"Decission_trees"] = decision_tree 
    #model_dictionary[f"Random_Forest"] = randomforest 
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
    # Transform the test labels (if needed)
    y_test_encoded = label_encoder.transform(y_test) 

    # Cross Gradient Booster
    xgb = XGBClassifier(n_estimators=1000, learning_rate=0.05)
    fit_and_save_models_xgboost(xgb, X_train, y_train_encoded,y_test_encoded,"Cross_Gradient_Booster")

    # Cross Gradient Booster (Random Forest)
    xgbrf = XGBRFClassifier(objective= 'multi:softmax')
    fit_and_save_models_xgboost(xgbrf, X_train, y_train_encoded,y_test_encoded,"Cross_Gradient_Booster_Random_Forest")

    #print(y_train_encoded)    


    




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
   
