import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Sample data
data = {
    'House Size (sq ft)': [750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200],
    'Number of Bedrooms': [2, 2, 3, 3, 3, 4, 4, 4, 5, 5],
    'House Age (years)': [20, 15, 15, 10, 10, 5, 5, 2, 2, 1],
    'House Price ($1000)': [150, 160, 165, 175, 180, 195, 200, 210, 220, 230]
}

df = pd.DataFrame(data)

# Features and target variable
x = df[['House Size (sq ft)', 'Number of Bedrooms', 'House Age (years)']]
y = df['House Price ($1000)']

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Linear Regression
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)

# Predictions for Linear Regression
y_pred_linear = linear_model.predict(x_test)

# Polynomial Regression
poly = PolynomialFeatures(degree=2)  # You can change the degree as needed
x_poly_train = poly.fit_transform(x_train)
x_poly_test = poly.transform(x_test)

# Train the polynomial regression model
poly_model = LinearRegression()
poly_model.fit(x_poly_train, y_train)

# Predictions for Polynomial Regression
y_pred_poly = poly_model.predict(x_poly_test)

# Plotting
plt.plot(x_test['House Size (sq ft)'], y_test, color='blue', label='Actual price')
plt.plot(x_test['House Size (sq ft)'], y_pred_linear, color='red', label='Predicted price (Linear)', marker='x')
plt.plot(x_test['House Size (sq ft)'], y_pred_poly, color='green', label='Predicted price (Polynomial)', marker='o')
plt.xlabel('House Size (sq ft)')
plt.ylabel('House Price ($1000)')
plt.title('House Price Prediction using Linear and Polynomial Regression')
plt.legend()
plt.show()

# Accuracy
accuracy_linear = linear_model.score(x_test, y_test)
accuracy_poly = poly_model.score(x_poly_test, y_test)

print(f"Linear model accuracy: {accuracy_linear * 100:.2f}%")
print(f"Polynomial model accuracy: {accuracy_poly * 100:.2f}%")
