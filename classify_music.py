import librosa
import numpy as np
import json
import os
from keras.models import load_model
import joblib

def extract_features(file_path):
    try:
        audio, _ = librosa.load(file_path, sr=22050, mono=True)
        mfccs = librosa.feature.mfcc(y=audio, sr=22050, n_mfcc=13)
        chroma = librosa.feature.chroma_stft(y=audio, sr=22050)
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=22050)
        tonnetz = librosa.feature.tonnetz(y=audio, sr=22050)
        features = np.vstack([mfccs, chroma, spectral_contrast, tonnetz])
        return np.mean(features.T, axis=0)
    except Exception as e:
        print(f"Error encountered while parsing file '{file_path}': {e}")
        return None
    
def classify_music(file_path):
    saved_folder = 'model_folder'
    # Load data and models from the saved folder
    # y_test = np.load(os.path.join(saved_folder, 'y_test.npy'))
    cnn_model = load_model(os.path.join(saved_folder, 'cnn_model.h5'))
    baseline_cnn_model = load_model(os.path.join(saved_folder, 'baseline_cnn_model.h5'))
    rf_model = joblib.load(os.path.join(saved_folder, 'rf_model.joblib'))
    svm_model = joblib.load(os.path.join(saved_folder, 'svm_model.joblib'))
    xgb_model = joblib.load(os.path.join(saved_folder, 'xgb_model.joblib'))
    # ensemble_input = np.load(os.path.join(saved_folder, 'ensemble_input.npy'))
    # Load label dictionary
    with open(os.path.join(saved_folder, 'label_dict.json'), 'r') as json_file:
        label_dict = json.load(json_file)
    # Extract features
    sample_features = extract_features(file_path)
    sample_features_cnn = sample_features.reshape(1, sample_features.shape[0], 1)
    # Predict with baseline CNN model
    baseline_cnn_prediction_prob = baseline_cnn_model.predict(sample_features_cnn)[0]
    genre_names = {idx: genre for genre, idx in label_dict.items()}
    predicted_percentages_baseline_cnn = {genre_names[idx]: float(prob) * 100 for idx, prob in enumerate(baseline_cnn_prediction_prob)}
    predicted_genre_baseline_cnn = max(predicted_percentages_baseline_cnn, key=predicted_percentages_baseline_cnn.get)
    sorted_baseline_cnn_predictions = dict(sorted(predicted_percentages_baseline_cnn.items(), key=lambda item: item[1], reverse=True))
    sorted_json_result_baseline_cnn = {
        "Predicted Genre (Baseline CNN)": predicted_genre_baseline_cnn,
        "Predicted Percentages (Baseline CNN)": sorted_baseline_cnn_predictions
    }
    cnn_prediction_prob = cnn_model.predict(sample_features_cnn)[0]
    # Predict with SVM, Random Forest, and XGBoost models
    cnn_features_sample = cnn_model.predict(sample_features_cnn)
    svm_prediction_sample = svm_model.predict(cnn_features_sample)
    rf_prediction_sample = rf_model.predict(cnn_features_sample)
    ensemble_input_sample = np.column_stack((svm_prediction_sample, rf_prediction_sample, np.argmax(cnn_prediction_prob)))
    ensemble_prediction_sample = xgb_model.predict(ensemble_input_sample)
    # Organize predictions
    predicted_genre_ensemble = genre_names[ensemble_prediction_sample[0]]
    predicted_probabilities_ensemble = xgb_model.predict_proba(ensemble_input_sample)[0]
    predicted_percentages_ensemble = {genre_names[idx]: float(prob) * 100 for idx, prob in enumerate(predicted_probabilities_ensemble)}
    sorted_ensemble_predictions = dict(sorted(predicted_percentages_ensemble.items(), key=lambda item: item[1], reverse=True))
    sorted_json_result_ensemble = {
        "Predicted Genre (OCNN)": predicted_genre_ensemble,
        "Predicted Percentages (OCNN)": sorted_ensemble_predictions
    }
    # Merge results
    merged_json_result = {"Baseline CNN": sorted_json_result_baseline_cnn, "OCNN": sorted_json_result_ensemble}
    
    return merged_json_result