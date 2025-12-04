import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

height = np.array([150,155,160,165,170,175,180]).reshape(-1,1)
weight = np.array([50,55,60,63,68,72,75])

model_no = LinearRegression()
model_no.fit(height, weight)

slope_no = model_no.coef_[0]
intercept_no = model_no.intercept_
r2_no = r2_score(weight, model_no.predict(height))

print("WITHOUT outlier:")
print("Slope:", slope_no)
print("Intercept:", intercept_no)
print("R²:", r2_no)

height_out = np.append(height, [[190]], axis=0)
weight_out = np.append(weight, 60)

model_out = LinearRegression()
model_out.fit(height_out, weight_out)

slope_out = model_out.coef_[0]
intercept_out = model_out.intercept_
r2_out = r2_score(weight_out, model_out.predict(height_out))

print("\nWITH outlier:")
print("Slope:", slope_out)
print("Intercept:", intercept_out)
print("R²:", r2_out)
