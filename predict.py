# import pickle
import os
import pandas as pd

import mlflow
from flask import Flask, request, jsonify

RUN_ID = os.getenv('RUN_ID')

# logged_model = f'/home/ubuntu/wine_classification/mlruns/2/{RUN_ID}/artifacts/model'
# logged_model = f'/home/ubuntu/wine_classification/model'
# this is for running in docker
logged_model = f'/app/model'
model = mlflow.pyfunc.load_model(logged_model)

def prepare_features(wine):
    wine_df = pd.DataFrame([wine])

    # Acidity Balance
    wine_df['total_acidity'] = wine_df['fixed acidity'] + wine_df['volatile acidity'] + wine_df['citric acid']
    # Sulfur Dioxide Ratio
    wine_df['so2_ratio'] = wine_df['free sulfur dioxide'] / wine_df['total sulfur dioxide']
    # Sugar to Alcohol Ratio
    wine_df['sugar_alcohol_ratio'] = wine_df['residual sugar'] / wine_df['alcohol']
    # Interaction Terms
    wine_df['density_alcohol_interaction'] = wine_df['density'] * wine_df['alcohol']
    # Sulphates to Chlorides Ratio
    wine_df['sulphates_chlorides_ratio'] = wine_df['sulphates'] / wine_df['chlorides']

    selected_features = ['volatile acidity', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'total_acidity',
                        'so2_ratio', 'sugar_alcohol_ratio', 'density_alcohol_interaction', 'sulphates_chlorides_ratio']

    X = wine_df[selected_features].values  # Convert DataFrame to numpy array

    return X

def predict(features):
    pred = model.predict(features)
    return pred

app = Flask('wine-classification')

@app.route('/classification', methods=['POST'])
def classification_endpoint():
    wine = request.get_json()

    features = prepare_features(wine)
    pred = predict(features)

    result = {
        'features': wine,
        'quality': pred.tolist(),
        'model_version': RUN_ID
    }
    res = jsonify(result)
    return res

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
