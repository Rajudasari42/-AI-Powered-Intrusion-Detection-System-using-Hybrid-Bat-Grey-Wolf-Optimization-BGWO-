import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def bgwo_feature_selection(X, y, n_iterations=15, n_features=15):
    np.random.seed(42)
    feature_count = X.shape[1]
    best_features = np.random.choice(feature_count, n_features, replace=False)
    best_score = 0

    for i in range(n_iterations):
        candidate = np.random.choice(feature_count, n_features, replace=False)
        X_subset = X.iloc[:, candidate]
        X_train, X_test, y_train, y_test = train_test_split(X_subset, y, test_size=0.2, random_state=42)
        model = RandomForestClassifier()
        model.fit(X_train, y_train)
        score = accuracy_score(y_test, model.predict(X_test))

        if score > best_score:
            best_score = score
            best_features = candidate

        print(f"Iteration {i+1}/{n_iterations}, Best Score: {best_score:.4f}")

    return best_features
