import joblib
from preprocess import load_dataset
from bgwo import bgwo_feature_selection
from sklearn.ensemble import RandomForestClassifier

X, y = load_dataset("data/NSL_Binary.csv")

selected_indices = bgwo_feature_selection(X, y, n_iterations=15, n_features=15)
X_selected = X.iloc[:, selected_indices]

model = RandomForestClassifier()
model.fit(X_selected, y)

joblib.dump(model, "model/rf_model.pkl")
joblib.dump(selected_indices, "model/selected_features.pkl")

print("✅ Model and selected features saved!")
