from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)  # Use default static and templates folder

# Debug info
print("Working Directory:", os.getcwd())
print("Static exists:", os.path.exists('static'))
print("Templates exists:", os.path.exists('templates'))

# Load model (optional)
model = None
if os.path.exists('model.pkl'):
    try:
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
        print("Model loaded successfully!")
    except Exception as e:
        print("Error loading model:", e)
else:
    print("No model found, using dummy prediction.")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect form data
        age = int(request.form['age'])
        sex = 1 if request.form['sex'].lower() == 'male' else 0
        bmi = float(request.form['bmi'])
        children = int(request.form['children'])
        smoker = 1 if request.form['smoker'].lower() == 'yes' else 0

        region_map = {'southwest': 0, 'southeast': 1, 'northwest': 2, 'northeast': 3}
        region = region_map.get(request.form['region'].lower(), 0)

        features = [age, sex, bmi, children, smoker, region]

        # Prediction
        if model:
            # Assuming model is a dict with coefficients
            coeffs = model.get('coefficients', {})
            feature_names = model.get('feature_names', [])
            
            prediction = coeffs.get('intercept', 0)
            for i, feature_name in enumerate(feature_names):
                prediction += features[i] * coeffs.get(feature_name, 0)
        else:
            # Dummy formula
            prediction = 1200 + (age * 85) + (bmi * 120) + (children * 380) + (smoker * 7800) + (region * 280)
            prediction = max(1000, prediction)

        return render_template('index.html', prediction_text=f"${prediction:,.2f}")

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {e}")

if __name__ == "__main__":
    app.run(debug=True)
