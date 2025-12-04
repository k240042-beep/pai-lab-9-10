import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score
import matplotlib.pyplot as plt

df = pd.read_csv("train.csv")

df = df[["Survived", "Pclass", "Sex", "Age", "Fare"]]
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

X = df[["Pclass", "Sex", "Age", "Fare"]]
y = df["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(max_depth=4, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))

plt.figure(figsize=(14, 8))
plot_tree(
    model,
    feature_names=["Pclass", "Sex", "Age", "Fare"],
    class_names=["Died", "Survived"],
    filled=True,
    rounded=True
)
plt.show()
