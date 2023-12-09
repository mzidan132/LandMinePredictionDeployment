from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the trained model
with open('model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Load the scaler
with open('standscaler.pkl', 'rb') as scaler_file:
    loaded_scaler = pickle.load(scaler_file)

# Define a dictionary to map numerical categories to their names
category_mapping = {
    1: "Not-Mine",
    2: "Anti-tank",
    3: "Anti-personnel",
    4: "Booby-Trapped-Anti-personnel",
    5: "M14-Anti-personnel"
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input values from the form
    voltage_input = float(request.form['Voltage'])
    height_input = float(request.form['Height'])
    soil_input = float(request.form['Soil'])

    # Create a numpy array with the input features
    new_data = np.array([[voltage_input, height_input, soil_input]])

    # Scale the new data using the loaded scaler
    scaled_data = loaded_scaler.transform(new_data)

    # Make predictions using the loaded model
    prediction = loaded_model.predict(scaled_data)

    # Map numerical category to its name
    predicted_category = category_mapping[prediction[0]]

    return render_template('index.html', prediction=predicted_category)

if __name__ == '__main__':
    app.run(debug=True)
