import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib

data = pd.read_csv('soil_data.csv')  # temperature, humidity, pH, moisture

X = data[['temperature', 'humidity', 'pH']]
y = data['moisture']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestRegressor(n_estimators=100)
model.fit(X_train, y_train)

joblib.dump(model, 'moisture_predictor.pkl')
