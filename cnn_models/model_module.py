import os
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from keras.models import Sequential, load_model
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from keras.utils import to_categorical
import librosa
from pydub import AudioSegment
import joblib
import json
def extract_features(file_path):
    try:
        # Load MP3 file and convert to WAV
        # audio = AudioSegment.from_mp3(file_path)
        # audio = audio.set_channels(1)  # Convert stereo to mono
        # audio.export("temp.wav", format="wav")
        audio, _ = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=22050)
        features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
        mean_features = np.mean(features.T, axis=0)
        return mean_features
    except Exception as e:
        print(f"Error encountered while parsing file '{file_path}': {e}")
        return None
# Load label_dict
with open('./label_dict.json', 'r') as json_file:
    label_dict = json.load(json_file)

# Load X_test
X_test = np.load('X_test.npy')
X_test_cnn = np.load('X_test_cnn.npy')
# Load models
def optimize_cnn_model(testing_feature, svm_classifier_optimized, rf_classifier_optimized, improved_cnn_model, normalized_weights, label_dict):
    result_dict = {}
    # Reshape features for CNN input
    testing_feature_cnn = testing_feature.reshape(1, testing_feature.shape[0], 1)
    # Make predictions with the individual models
    svm_prediction = svm_classifier_optimized.predict(testing_feature.reshape(1, -1))
    rf_prediction = rf_classifier_optimized.predict(testing_feature.reshape(1, -1))
    cnn_prediction_probs = improved_cnn_model.predict(testing_feature_cnn)
    cnn_prediction = np.argmax(cnn_prediction_probs, axis=1)
    # Ensemble: Weighted Voting with normalized weights
    ensemble_prediction_probs = (
        normalized_weights[0] * to_categorical(svm_prediction, num_classes=len(label_dict)) +
        normalized_weights[1] * to_categorical(rf_prediction, num_classes=len(label_dict)) +
        normalized_weights[2] * cnn_prediction_probs
    )
    # Normalize ensemble predictions to ensure they sum up to 1
    normalized_ensemble_probs = ensemble_prediction_probs / sum(ensemble_prediction_probs[0])
    weighted_majority_voting_prediction = np.argmax(normalized_ensemble_probs)
    predicted_genre = list(label_dict.keys())[weighted_majority_voting_prediction]
    result_dict["predicted_genre"] = predicted_genre
    # Store the predicted percentages in the result dictionary
    result_dict["predicted_percentages"] = {genre: percentage.item() * 100 for genre, percentage in zip(label_dict.keys(), normalized_ensemble_probs[0])}
    return json.dumps(result_dict, indent=2)

def evaluate_cnn_model(testing_feature, loaded_model, label_dict):
    result_dict = {}
    # Check if testing_feature is not None
    if testing_feature is not None:
        print(f"Shape of extracted features: {testing_feature.shape}")
        # Reshape features for CNN input
        testing_feature_cnn = testing_feature.reshape(1, testing_feature.shape[0], 1)
        # Use the model to predict the genre
        prediction = loaded_model.predict(testing_feature_cnn)
        # Get the predicted percentages for each genre
        predicted_percentages = (prediction * 100).tolist()[0]
        # Create a list of tuples with genre and its percentage
        genre_percentage_list = [(genre, percentage) for genre, percentage in zip(label_dict.keys(), predicted_percentages)]
        # Sort the list based on percentage in descending order
        genre_percentage_list.sort(key=lambda x: x[1], reverse=True)
        # Store the predicted genre and percentage in the result dictionary
        result_dict["predicted_genre"] = genre_percentage_list[0][0]
        result_dict["predicted_percentages"] = {genre: percentage for genre, percentage in genre_percentage_list}
    else:
        result_dict["error_message"] = "Error extracting features from 'testing.wav'"

    return json.dumps(result_dict, indent=2)


svm_classifier_optimized = joblib.load('optimized_svm_model.joblib')
rf_classifier_optimized = joblib.load('optimized_rf_model.joblib')
improved_cnn_model = load_model('improved_cnn_model.h5')

# Make predictions with the individual models
svm_predictions_optimized = svm_classifier_optimized.predict(X_test.reshape(X_test.shape[0], -1))
rf_predictions_optimized = rf_classifier_optimized.predict(X_test.reshape(X_test.shape[0], -1))
cnn_predictions_optimized_probs = improved_cnn_model.predict(X_test_cnn)
cnn_predictions_optimized = np.argmax(cnn_predictions_optimized_probs, axis=1)
# Ensemble: Weighted Voting
weights = [0.2, 0.2, 0.6]  # Adjust these weights based on individual model performance
# Normalize weights to ensure they sum up to 1
normalized_weights = np.array(weights) / sum(weights)

# Extract features from testing.wav
testing_file_path = './testing.wav'  # Replace with the actual path
testing_feature = extract_features(testing_file_path)

result_optimize_cnn = optimize_cnn_model(testing_feature, svm_classifier_optimized, rf_classifier_optimized, improved_cnn_model, normalized_weights, label_dict)
result_cnn = evaluate_cnn_model(testing_feature, improved_cnn_model, label_dict)
print(result_optimize_cnn)
print(result_cnn)




