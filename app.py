from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

app = Flask(__name__)

# Load the trained model if it exists
try:
    with open('polynomial_regression_model.pkl', 'rb') as f:
        model = pickle.load(f)
        poly = pickle.load(f)  # Load the polynomial features
except FileNotFoundError:
    model = None
    poly = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/train', methods=['POST'])
def train():
    global model, poly
    # Get the file from the request
    file = request.files['file']
    df = pd.read_csv(file)
    
    # Prepare the data
    X = df.iloc[:, 0:2].values  # CGPA and IQ as features
    y = df.iloc[:, -1].values  # LPA as the target

    # Polynomial transformation
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=2)
    
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # Save the model and polynomial features
    with open('polynomial_regression_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        pickle.dump(poly, f)

    # Test the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return jsonify({'message': 'Model trained successfully!', 'mse': mse, 'r2_score': r2})

@app.route('/predict', methods=['POST'])
def predict():
    global model, poly
    if not model:
        return jsonify({'error': 'Model not trained yet. Please upload a dataset and train the model first.'})

    # Get the input values from the request
    cgpa = float(request.form['cgpa'])
    iq = float(request.form['iq'])

    # Polynomial transformation of the inputs
    X_poly = poly.transform(np.array([[cgpa, iq]]))
    
    # Predict the LPA based on the input CGPA and IQ
    lpa = model.predict(X_poly)[0]
    
    return jsonify({'cgpa': cgpa, 'iq': iq, 'predicted_lpa': lpa})

if __name__ == '__main__':
    app.run(debug=True)
