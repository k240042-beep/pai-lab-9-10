import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

np.random.seed(42)

screen_on = np.random.normal(5, 1.2, 100)
battery_life = 20 - 1.8*screen_on + np.random.normal(0, 1.5, 100)

df = pd.DataFrame({
    "ScreenOnTime": screen_on,
    "BatteryLife": battery_life
})

corr, p_val = pearsonr(df["ScreenOnTime"], df["BatteryLife"])
print("Correlation:", corr)
print("p-value:", p_val)

X = df[["ScreenOnTime"]]
y = df["BatteryLife"]

model = LinearRegression()
model.fit(X, y)

print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

pred = model.predict([[6]])
print("Predicted battery life for 6 hours SOT:", pred[0])

r2 = model.score(X, y)
print("R²:", r2)
