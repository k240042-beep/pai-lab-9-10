import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("laptops.csv")   

df = df[['Price', 'RAM', 'Weight', 'CPU Frequency', 'Storage']]

df = df.dropna()

X = df[['RAM', 'Weight', 'CPU Frequency', 'Storage']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model1 = LinearRegression()
model1.fit(X_train, y_train)
pred1 = model1.predict(X_test)

print("MODEL 1 — WITHOUT SCALING")
print("R² Score:", r2_score(y_test, pred1))
print("MSE:", mean_squared_error(y_test, pred1))
print("Coefficients:", model1.coef_)
print()

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model2 = LinearRegression()
model2.fit(X_train_scaled, y_train)
pred2 = model2.predict(X_test_scaled)

print("MODEL 2 — WITH SCALING")
print("R² Score:", r2_score(y_test, pred2))
print("MSE:", mean_squared_error(y_test, pred2))
print("Coefficients:", model2.coef_)
print()

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': abs(model2.coef_)  
})

coef_df = coef_df.sort_values(by='Coefficient', ascending=False)

print("Feature Importance (Most Significant Predictor First):")
print(coef_df)
