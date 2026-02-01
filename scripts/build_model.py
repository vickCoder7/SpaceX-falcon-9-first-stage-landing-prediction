import pandas as pd
import numpy as np
import pickle
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

print("Loading data...")
# Load Features
X_df = pd.read_csv('../data/features_one_hot.csv')
feature_names = X_df.columns.tolist()
X = X_df.values

# Load Target
df_target = pd.read_csv('../data/spacexdata.csv')
y = df_target['Class'].to_numpy()

print(f"Features shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Standardize
print("Scaling data...")
scaler = preprocessing.StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression
# Parameters: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
print("Training Logistic Regression...")
model = LogisticRegression(C=0.1, penalty='l2', solver='lbfgs')
model.fit(X_scaled, y)

# Evaluate on full set just to be sure
score = model.score(X_scaled, y)
print(f"Model Accuracy on full dataset: {score:.4f}")

# Save Everything
artifact = {
    'model': model,
    'scaler': scaler,
    'feature_names': feature_names
}

with open('../data/model.pkl', 'wb') as f:
    pickle.dump(artifact, f)

print("Model saved to ../data/model.pkl")
