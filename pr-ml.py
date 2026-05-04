import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# Load dataset
data = pd.read_csv(r"C:\Users\Admin\OneDrive\Desktop\python\projects\std.csv")

# Features & target
X = data[['hours']]
y = data['score']

# Split data (important for real ML)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = np.sqrt(mse)

print("\n Model Evaluation:")
print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)

# User input prediction
hours = float(input("\nEnter study hours: "))
predicted_score = model.predict([[hours]])

print(f"Predicted Score: {predicted_score[0]:.2f}")


plt.scatter(X, y, color='blue')
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Study Hours")
plt.ylabel("Score")
plt.title("Study Hours vs Score Prediction")
plt.show()