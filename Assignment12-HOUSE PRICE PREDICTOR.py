'''Assignment (09/03/2026)
Assignment Name : House Price Predictor
Description : Train a Linear Regression model, predict prices, and test with new input.'''

import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a sample dataset
data = {
    "Size_sqft": [500, 800, 1000, 1200, 1500, 1800, 2000],
    "Price": [100000, 150000, 200000, 230000, 300000, 350000, 400000]
}

# Convert to DataFrame
dataset = pd.DataFrame(data)

print("Dataset:")
print(dataset)

# Features (input)
X = dataset[["Size_sqft"]]

# Labels (output)
y = dataset["Price"]

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict price for a new house size
new_size = [[1600]]
predicted_price = model.predict(new_size)

print("\nPredicted price for house size 1600 sqft:", predicted_price[0])