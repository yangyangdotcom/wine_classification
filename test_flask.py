import requests

ride = {
    "fixed acidity": 7.4,
    "volatile acidity": 0.25,
    "citric acid": 0.29,
    "residual sugar": 2.2,
    "chlorides": 0.054,
    "free sulfur dioxide": 19.0,
    "total sulfur dioxide": 49.0,
    "density": 0.99666,
    "pH": 3.4,
    "sulphates" : 0.76,
    "alcohol": 10.9,
}

url = 'http://localhost:9696/classification'
response = requests.post(url, json=ride)
print(response.json())