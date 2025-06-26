import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Step 1: Download Stock Data
data = yf.download('AAPL', start='2020-01-01', end='2024-12-31')

# Step 2: Prepare Data
data = data[['Close']]
data['Prediction'] = data['Close'].shift(-30)  # Predict 30 days ahead

# Step 3: Features and Labels
X = np.array(data.drop(['Prediction'], axis=1))[:-30]
y = np.array(data['Prediction'])[:-30]

# Step 4: Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Step 5: Train the Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 6: Predictions
predictions = model.predict(X_test)

# Step 7: Plot the results
plt.plot(y_test, label='Actual')
plt.plot(predictions, label='Predicted')
plt.legend()
plt.title("Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.show()
