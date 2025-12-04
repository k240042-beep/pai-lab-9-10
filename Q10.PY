import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

df = pd.read_csv("data.csv", encoding='latin-1')   



df = df[['Value', 'Overall', 'Potential', 'Age', 'International Reputation']]

def convert_value(x):
    x = x.replace('€','')
    if 'M' in x:
        return float(x.replace('M','')) * 1_000_000
    elif 'K' in x:
        return float(x.replace('K','')) * 1_000
    else:
        return float(x)

df['Value'] = df['Value'].apply(convert_value)

df = df.dropna()

X1 = df[['Overall', 'Potential', 'Age', 'International Reputation']]
y = df['Value']

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.2, random_state=42)

model1 = LinearRegression()
model1.fit(X_train, y_train)

y_pred1 = model1.predict(X_test)

print("MODEL 1 (WITH Potential)")
print("Coefficients:", model1.coef_)
print("R² Score:", r2_score(y_test, y_pred1))
print("MSE:", mean_squared_error(y_test, y_pred1))
print()

coef_df = pd.DataFrame({
    'Feature': X1.columns,
    'Coefficient': model1.coef_
})
print("Impact of Features on Value:")
print(coef_df.sort_values(by='Coefficient', ascending=False))
print()

X2 = df[['Overall', 'Age', 'International Reputation']]

X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

model2 = LinearRegression()
model2.fit(X_train, y_train)

y_pred2 = model2.predict(X_test)

print("MODEL 2 (WITHOUT Potential)")
print("Coefficients:", model2.coef_)
print("R² Score:", r2_score(y_test, y_pred2))
print("MSE:", mean_squared_error(y_test, y_pred2))
