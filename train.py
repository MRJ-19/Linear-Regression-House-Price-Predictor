from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# Training data (house size vs. price)
X = np.array([[500], [1000], [1500], [2000]])   # 2D array
y = np.array([150, 200, 250, 300])              # 1D array

# Train model
model = LinearRegression()
model.fit(X, y)

# ----- Custom Input -----
size = float(input("Enter house size in sq.ft: "))
prediction = model.predict([[size]])
print(f"Predicted price for {size} sq.ft = {prediction[0]:.2f} ($1000s)")

# ---- Visualization ----
plt.scatter(X, y, color="blue", label="Data points")

# Regression line
X_line = np.linspace(400, 2100, 100).reshape(-1, 1)
y_line = model.predict(X_line)
plt.plot(X_line, y_line, color="red", label="Regression line")

# Prediction point
plt.scatter([size], prediction, color="green", marker="x", s=100, label=f"Prediction ({size} sq.ft)")

# Labels and legend
plt.xlabel("House size (sq.ft)")
plt.ylabel("Price ($1000s)")
plt.title("Linear Regression Example")
plt.legend()
plt.show()
