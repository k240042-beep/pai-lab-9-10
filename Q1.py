import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X = np.array([[1.1],[1.3],[1.5],[2.0],[2.2],[2.9],[3.0],[3.2],[3.2],[3.7]])
Y = np.array([39.0,46.0,47.0,52.0,56.0,64.0,65.0,67.0,68.0,70.0])

model = LinearRegression()
model.fit(X, Y)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

pred = model.predict([[4.5]])
print("Prediction for 4.5 years:", pred[0])

r2 = r2_score(Y, model.predict(X))
print("R²:", r2)


#SCRATCH
import numpy as np

X = np.array([1.1, 1.3, 1.5, 2.0, 2.2, 2.9, 3.0, 3.2, 3.2, 3.7])
Y = np.array([39.0, 46.0, 47.0, 52.0, 56.0, 64.0, 65.0, 67.0, 68.0, 70.0])

x_mean = np.mean(X)
y_mean = np.mean(Y)

b1 = np.sum((X - x_mean)*(Y - y_mean)) / np.sum((X - x_mean)**2)
b0 = y_mean - b1*x_mean

print("Intercept:", b0)
print("Slope:", b1)

pred_salary = b0 + b1*4.5
print("Prediction for 4.5 years:", pred_salary)

Y_pred = b0 + b1*X
r2 = 1 - (np.sum((Y - Y_pred)**2) / np.sum((Y - y_mean)**2))
print("R²:", r2)
