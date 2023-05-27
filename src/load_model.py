import mlflow

from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes

db = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(db.data, db.target)

model = mlflow.sklearn.load_model("runs:/d7ade5106ee341e0b4c63a53a9776231")
predictions = model.predict(X_test)
print(predictions)