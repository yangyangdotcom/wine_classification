import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import SMOTE
import mlflow
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("wine-classification-experiment")

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
Apply SMOTE
'''
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

def objective(params):
    with mlflow.start_run():
        mlflow.set_tag("developer", "bnai")
        # Log parameters
        mlflow.log_params(params)

        model = Sequential([
            Dense(params['units1'], activation=params['activation'], kernel_regularizer=l2(params['l2'])),
            Dropout(params['dropout']),
            Dense(params['units2'], activation=params['activation'], kernel_regularizer=l2(params['l2'])),
            Dropout(params['dropout']),
            Dense(params['units3'], activation=params['activation'], kernel_regularizer=l2(params['l2'])),
            Dense(y_train_smote.shape[1], activation='softmax')
        ])

        model.compile(optimizer=params['optimizer'],
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        
        model.fit(X_train_smote, y_train_smote, epochs=50, batch_size=int(params['batch_size']),
                  validation_split=0.2, verbose=1)
        
        test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
        
        # Log metrics
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_loss", test_loss)
        
        # Here we aim to minimize loss, hence return the loss
        return {'loss': -test_acc, 'status': STATUS_OK, 'model': model}

space = {
    'units1': scope.int(hp.quniform('units1', 50, 150, 1)),
    'units2': scope.int(hp.quniform('units2', 30, 120, 1)),
    'units3': scope.int(hp.quniform('units3', 20, 100, 1)),
    'dropout': hp.uniform('dropout', 0.1, 0.5),
    'activation': hp.choice('activation', ['relu', 'tanh']),
    'l2': hp.loguniform('l2', np.log(0.0001), np.log(0.01)),
    'optimizer': hp.choice('optimizer', ['adam', 'sgd']),
    'batch_size': hp.quniform('batch_size', 16, 64, 1)
}

trials = Trials()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=1,
            trials=trials)

print(best)