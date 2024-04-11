import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import numpy as np
df = pd.read_csv("data.csv")

"""
Data manipulation
"""
# Acidity Balance
df['total_acidity'] = df['fixed acidity'] + df['volatile acidity'] + df['citric acid']

# Sulfur Dioxide Ratio
df['so2_ratio'] = df['free sulfur dioxide'] / df['total sulfur dioxide']

# Sugar to Alcohol Ratio
df['sugar_alcohol_ratio'] = df['residual sugar'] / df['alcohol']

# Interaction Terms
df['density_alcohol_interaction'] = df['density'] * df['alcohol']

# Sulphates to Chlorides Ratio
df['sulphates_chlorides_ratio'] = df['sulphates'] / df['chlorides']

# show distribution
target = ['quality']
plt.hist(df[target], bins=20, alpha=0.7)
plt.xlabel(target)
plt.ylabel('Frequency')
plt.title(f'Frequency distribution of {target}')
# plt.show()

# Split dataset
y = df[target]
X = df.drop(columns=target)

'''
Normalization
- In this case normalization is not needed
'''
# Normalize data
# scalar = StandardScaler()
# X_scaled = scalar.fit_transform(X)

# Split dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(np.unique(y_train))
'''
Apply SMOTE
- In this case SMOTE is not needed
'''
# smote = SMOTE(random_state=42)
# X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Determine important features
model = RandomForestClassifier(random_state=42)

rfe = RFE(estimator=model, n_features_to_select=5)

# rfe.fit(X_train_smote, y_train_smote.values.ravel())
rfe.fit(X_train, y_train.values.ravel())

# -- Transform data based on RFE
# X_train_rfe = rfe.transform(X_train_smote)
X_train_rfe = rfe.transform(X_train)
X_test_rfe = rfe.transform(X_test)

# -- Train and evaluate model using selected features
# model.fit(X_train_rfe, y_train_smote.values.ravel())
model.fit(X_train_rfe, y_train.values.ravel())
y_pred = model.predict(X_test_rfe)
# print(y_pred)

# -- calculate the performance metric
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# -- see selected features
features = pd.Series(rfe.support_, index=X.columns)
selected_features = features[features == True].index.tolist()
print("Selected features:", selected_features)

'''
Important features = ['volatile acidity', 'chlorides', 'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol', 'total_acidity',
                        'so2_ratio', 'sugar_alcohol_ratio', 'density_alcohol_interaction', 'sulphates_chlorides_ratio']
'''