from flask import Flask, request, render_template
import pickle
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load the trained model
model_path = 'fashion_price_predictor.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Home route to load the input form
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route that takes input and returns the predicted price
@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    product_name = request.form['product_name']
    brand = request.form['brand']
    category = request.form['category']
    color = request.form['color']
    size = request.form['size']

    # Prepare the input as a pandas DataFrame with the same structure as the training data
    input_data = pd.DataFrame([[product_name, brand, category, color, size]],
                              columns=['Product Name', 'Brand', 'Category', 'Color', 'Size'])

    # Use the loaded model to make a prediction
    predicted_price = model.predict(input_data)[0]

    # Return the result as a web page
    return render_template('index.html', prediction_text=f"Predicted Price: ${predicted_price:.2f}")

if __name__ == "__main__":
    app.run(debug=True)
