import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

height = np.array([150,155,160,165,170,175,180]).reshape(-1,1)
weight = np.array([50,55,60,63,68,72,75])

model = LinearRegression()
model.fit(height, weight)

intercept = model.intercept_
slope = model.coef_[0]

print("Intercept:", intercept)
print("Slope:", slope)

pred_172 = model.predict([[172]])[0]
print("Predicted weight for 172 cm:", pred_172)

predictions = model.predict(height)
residuals = weight - predictions
print("Residuals:", residuals)

plt.scatter(height, weight, label="Data Points")
plt.plot(height, predictions, label="Regression Line")
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.title("Height vs Weight")
plt.legend()
plt.show()

plt.scatter(predictions, residuals)
plt.axhline(0, color='black', linewidth=1)
plt.xlabel("Fitted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.show()
