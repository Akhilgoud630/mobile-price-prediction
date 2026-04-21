from flask import Flask, request, render_template
import pickle
import pandas as pd
from sklearn import linear_model

app = Flask(__name__)

# Load the trained model
def load_model():
    try:
        with open("cellphone_price_model_v2.pkl", "rb") as f:
            model = pickle.load(f)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None

model = load_model()

# List of all features the model expects (from your training data)
REQUIRED_FEATURES = [
    'battery_power', 'blue', 'clock_speed', 'dual_sim', 'fc', 
    'four_g', 'int_memory', 'm_dep', 'mobile_wt', 'n_cores', 
    'pc', 'px_height', 'px_width', 'ram', 'sc_h', 'sc_w',
    'talk_time', 'three_g', 'touch_screen', 'wifi'
]

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        try:
            # Get all form data with default 0 for missing values
            input_data = {feature: float(request.form.get(feature, 0)) for feature in REQUIRED_FEATURES}
            
            # Convert to DataFrame with correct feature order
            input_df = pd.DataFrame([input_data], columns=REQUIRED_FEATURES)
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            
            # Map prediction to price range
            price_ranges = {
                0: "Low Cost",
                1: "Medium Cost",
                2: "High Cost",
                3: "Very High Cost"
            }
            
            result = price_ranges.get(prediction, "Unknown Range")
            return render_template('index.html', 
                                prediction_text=f'Predicted Price Range: {result}',
                                features=REQUIRED_FEATURES)
            
        except Exception as e:
            return render_template('index.html', 
                                prediction_text=f'Error: {str(e)}',
                                features=REQUIRED_FEATURES)
    
    return render_template('index.html', features=REQUIRED_FEATURES)

if __name__ == '__main__':
    app.run(debug=True)