import numpy as np
from keras.utils import to_categorical
import librosa
import json
def extract_features(file_path):
    try:
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
    return result_dict

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

    return result_dict

with open('./cnn_models/label_dict.json', 'r') as json_file:
    label_dict = json.load(json_file)

def classify_music(result_optimize_cnn, result_cnn):
    response = {
        "result_optimize_cnn": result_optimize_cnn,
        "result_cnn": result_cnn
    }
    return response