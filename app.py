from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

model = joblib.load('model/rf_model.pkl')
selected_indices = joblib.load('model/selected_features.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[f'feature{i}']) for i in range(41)]
        input_array = np.array(input_features).reshape(1, -1)
        input_selected = input_array[:, selected_indices]

        prediction = model.predict(input_selected)
        result = "Normal" if prediction[0] == 0 else "Attack"

        return f"<h2>Prediction Result: {result}</h2>"

    except Exception as e:
        return f"<h2>Error: {e}</h2>"

if __name__ == '__main__':
    app.run(debug=True)
