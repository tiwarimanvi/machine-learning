from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import numpy as np

digits = load_digits()
X = digits.data
y = digits.target

D = X.shape[1]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

correlation_coefficients = np.abs(np.corrcoef(X_scaled.T, y)[:D, -1])

avg_accuracies = []
for i in range(1, D+1):
    selected_features = np.argsort(correlation_coefficients)[-i:]
    X_selected = X_scaled[:, selected_features]
    avg_accuracy = 0
    for _ in range(10):
        X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2)
        gnb = GaussianNB()
        gnb.fit(X_train, y_train)
        y_pred = gnb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        avg_accuracy += accuracy
    avg_accuracy /= 10
    avg_accuracies.append(avg_accuracy)

num_features = len(avg_accuracies)
for i in range(num_features):
    acc = avg_accuracies[i]
    print("no of features", i+1)
    print("average accuracies", acc)
