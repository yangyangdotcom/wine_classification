import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

def read_data(path):
    data = pd.read_csv('data.csv')

    data['quality'] = data['quality'] - 3
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
    y = data[target]
    X = data[selected_features]

    return X, y

def encode_data(y):
    return to_categorical(y)


def train(X_train, y_train, num_class):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(num_class, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(X_train, y_train,
                        epochs=50,
                        batch_size = 32,
                        validation_split=0.2,
                        verbose = 1)

    return model, history

X, y = read_data('data.csv')
y_encoded = encode_data(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

model, history = train(X_train, y_train, len(np.unique(y['quality'])))

# test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
# print(f"Test accuracy: {test_acc}")
# print(f"Test loss: {test_loss}")