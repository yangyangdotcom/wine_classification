import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

import tensorflow as tf
from tensorflow.keras.utils import to_categorical

import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-classification-experiment-svc")
mlflow.sklearn.autolog()

def divide_wine_quality(data):
    bins = (2, 6.5, 8)
    group_names = ['bad', 'good']
    data['quality'] = pd.cut(data['quality'], bins = bins, labels = group_names)

    label_quality = LabelEncoder()
    data['quality'] = label_quality.fit_transform(data['quality'])
    return data

def read_data(path):
    data = pd.read_csv(path)

    target = ['quality']

    # Acidity Balance
    data['total_acidity'] =data['fixed acidity'] +data['volatile acidity'] +data['citric acid']
        # Sulfur Dioxide Ratio
    data['so2_ratio'] =data['free sulfur dioxide'] /data['total sulfur dioxide']
        # Sugar to Alcohol Ratio
    data['sugar_alcohol_ratio'] =data['residual sugar'] /data['alcohol']
        # Interaction Terms
    data['density_alcohol_interaction'] =data['density'] *data['alcohol']
        # Sulphates to Chlorides Ratio
    data['sulphates_chlorides_ratio'] =data['sulphates'] /data['chlorides']

    selected_features = ['volatile acidity', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'total_acidity',
                        'so2_ratio', 'sugar_alcohol_ratio', 'density_alcohol_interaction', 'sulphates_chlorides_ratio']
    data = divide_wine_quality(data)
    y = data[target]
    X = data[selected_features]

    return X, y

def encode_data(y):
    return to_categorical(y)

X, y = read_data('data.csv')
# y_encoded = encode_data(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

'''
Apply standard scalar
'''
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("developer", "bnai")

        svc = SVC(C = params['C'], gamma =  params['gamma'], kernel= params['kernel'])
        svc.fit(X_train, y_train)
        pred_svc = svc.predict(X_test)

        test_acc = accuracy_score(y_test, pred_svc)

        # Log parameters and metrics
        mlflow.log_params(params)
        mlflow.log_metric("accuracy", test_acc)

        return {'loss': -test_acc, 'status': STATUS_OK, 'model': svc}

space = {
    'C': hp.quniform('C', 0.1, 2, 0.1),
    'gamma': hp.quniform('gamma', 0.1, 2, 0.1),
    'kernel': hp.choice('kernel', ['linear', 'poly', 'rbf', 'sigmoid'])
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=100,
            trials=trials)

print(best)