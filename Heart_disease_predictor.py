import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler

# Load cleaned data
df = pd.read_csv("cleaned_heart_disease_data.csv")

# Check for missing values
if df.isnull().sum().any():
    print("Warning: Missing values found in the dataset.")
else:
    print("No missing values in the dataset.")

# Feature and Target Definition
X = df.drop("target", axis=1)  # Features
y = df["target"]  # Target variable

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scaling Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model Initialization
dt_classifier = DecisionTreeClassifier(random_state=42)

# Hyperparameter Tuning using GridSearchCV
param_grid = {
    'criterion': ['gini', 'entropy'],
    'max_depth': [None, 5, 10, 15, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4, 6]
}

grid_search = GridSearchCV(estimator=dt_classifier, param_grid=param_grid, 
                           scoring='accuracy', cv=5, n_jobs=-1, verbose=2)

# Fit the model with best hyperparameters
grid_search.fit(X_train, y_train)

# Best parameters and estimator
best_params = grid_search.best_params_
best_estimator = grid_search.best_estimator_

print("\nBest Hyperparameters:")
print(best_params)

# Model Evaluation
y_pred = best_estimator.predict(X_test)

print("\n--- Classification Report ---")
print(classification_report(y_test, y_pred))

print("\n--- Confusion Matrix ---")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

accuracy = accuracy_score(y_test, y_pred)
print(f"\nModel Accuracy: {accuracy * 100:.2f}%")

# Cross-validation to assess model stability
cv_scores = cross_val_score(best_estimator, X, y, cv=5)
print(f"\nCross-Validation Scores: {cv_scores}")
print(f"Mean CV Accuracy: {cv_scores.mean() * 100:.2f}%")

# Visualizing the Decision Tree
plt.figure(figsize=(15, 10))
plot_tree(best_estimator, filled=True, feature_names=X.columns, class_names=['No Heart Disease', 'Heart Disease'], rounded=True)
plt.title("Decision Tree Visualization")
plt.show()
