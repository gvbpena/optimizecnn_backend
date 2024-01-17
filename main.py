from fastapi import FastAPI
from classify_music import classify_music, extract_features, optimize_cnn_model, evaluate_cnn_model
import numpy as np
from keras.models import load_model
from fastapi import FastAPI, File, UploadFile

import joblib
import json
from fastapi.middleware.cors import CORSMiddleware
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Disable GPU usage

app = FastAPI()

origins = [ "http://localhost:3000","https://optimizedcnn.vercel.app/"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_methods=["*"],
    allow_credentials=True,
    allow_headers=["*"],
)
@app.post("/classify_music/")
async def classify_music_post(file: UploadFile = File(...)):
    with open('./cnn_models/label_dict.json', 'r') as json_file:
        label_dict = json.load(json_file)

    svm_classifier_optimized = joblib.load('./cnn_models/optimized_svm_model.joblib')
    rf_classifier_optimized = joblib.load('./cnn_models/optimized_rf_model.joblib')
    improved_cnn_model = load_model('./cnn_models/improved_cnn_model.h5')
    weights = [0.2, 0.2, 0.6]
    # Read the contents of the file
    contents = await file.read()
    # You can save the file if needed
    with open("uploaded_file.wav", "wb") as f:
        f.write(contents)

    # Extract features from the uploaded file
    testing_feature = extract_features("uploaded_file.wav")  # Replace with the actual path
    result_optimize_cnn = optimize_cnn_model(testing_feature, svm_classifier_optimized, rf_classifier_optimized, improved_cnn_model, weights, label_dict)
    result_cnn = evaluate_cnn_model(testing_feature, improved_cnn_model, label_dict)
    return classify_music(result_optimize_cnn, result_cnn)

@app.get("/classify_music/")
async def classify_music_endpoint():
    with open('./cnn_models/label_dict.json', 'r') as json_file:
        label_dict = json.load(json_file)

    # # Load X_test
    # X_test = np.load('./cnn_models/X_test.npy')
    # X_test_cnn = np.load('./cnn_models/X_test_cnn.npy')
    svm_classifier_optimized = joblib.load('./cnn_models/optimized_svm_model.joblib')
    rf_classifier_optimized = joblib.load('./cnn_models/optimized_rf_model.joblib')
    improved_cnn_model = load_model('./cnn_models/improved_cnn_model.h5')
    # Make predictions with the individual models
    # svm_predictions_optimized = svm_classifier_optimized.predict(X_test.reshape(X_test.shape[0], -1))
    # rf_predictions_optimized = rf_classifier_optimized.predict(X_test.reshape(X_test.shape[0], -1))
    # cnn_predictions_optimized_probs = improved_cnn_model.predict(X_test_cnn)
    # cnn_predictions_optimized = np.argmax(cnn_predictions_optimized_probs, axis=1)
    # Ensemble: Weighted Voting
    weights = [0.2, 0.2, 0.6]  # Adjust these weights based on individual model performance
    # Normalize weights to ensure they sum up to 1
    normalized_weights = np.array(weights) / sum(weights)
    # Extract features from testing.wav
    testing_file_path = './cnn_models/testing.wav'  # Replace with the actual path
    testing_feature = extract_features(testing_file_path)
    result_optimize_cnn = optimize_cnn_model(testing_feature, svm_classifier_optimized, rf_classifier_optimized, improved_cnn_model, normalized_weights, label_dict)
    result_cnn = evaluate_cnn_model(testing_feature, improved_cnn_model, label_dict)
    return classify_music(result_optimize_cnn, result_cnn)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)