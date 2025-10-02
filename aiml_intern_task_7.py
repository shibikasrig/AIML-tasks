

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

svm_linear = SVC(kernel="linear", C=1, random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_linear = svm_linear.predict(X_test)

svm_rbf = SVC(kernel="rbf", C=1, gamma="scale", random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_rbf = svm_rbf.predict(X_test)

print("Linear SVM Accuracy:", accuracy_score(y_test, y_pred_linear))
print("RBF SVM Accuracy:", accuracy_score(y_test, y_pred_rbf))

def plot_decision_boundary(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolor="k")
    plt.title(title)
    plt.show()

plot_decision_boundary(svm_linear, X_scaled, y, "Linear SVM Decision Boundary")
plot_decision_boundary(svm_rbf, X_scaled, y, "RBF SVM Decision Boundary")

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": ["scale", 0.1, 0.01, 0.001],
    "kernel": ["rbf"]
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring="accuracy")
grid_search.fit(X_scaled, y)

print("Best Hyperparameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

cv_linear = cross_val_score(svm_linear, X_scaled, y, cv=5)
cv_rbf = cross_val_score(svm_rbf, X_scaled, y, cv=5)

print("Linear SVM CV Accuracy (5-fold):", np.mean(cv_linear))
print("RBF SVM CV Accuracy (5-fold):", np.mean(cv_rbf))

data = load_breast_cancer()
X_bc, y_bc = data.data, data.target

X_bc_scaled = scaler.fit_transform(X_bc)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc_scaled, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

svm_bc = SVC(kernel="rbf", C=10, gamma=0.01, random_state=42)
svm_bc.fit(X_train_bc, y_train_bc)
y_pred_bc = svm_bc.predict(X_test_bc)

print("\nBreast Cancer Dataset Results:")
print("Accuracy:", accuracy_score(y_test_bc, y_pred_bc))
print("Classification Report:\n", classification_report(y_test_bc, y_pred_bc))
