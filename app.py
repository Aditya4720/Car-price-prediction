from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load ML model
with open('xgboost_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load LabelEncoder for 'name'
with open('name_encoding.pkl', 'rb') as f:
    le_name = pickle.load(f)
with open('fuel_encoding.pkl', 'rb') as f:
    le_fuel = pickle.load(f)
with open('seller_encoding.pkl', 'rb') as f:
    le_seller = pickle.load(f)
with open('transmission_encoding.pkl', 'rb') as f:
    le_trans = pickle.load(f)
with open('owner_encoding.pkl', 'rb') as f:
    le_owner = pickle.load(f)
    

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    try:
        print(request.form) 
        name = le_name.transform([request.form['name']])[0]
        fuel = le_fuel.transform([request.form['fuel']])[0]
        seller_type = le_seller.transform([request.form['seller_type']])[0]
        transmission = le_trans.transform([request.form['transmission']])[0]
        owner = le_owner.transform([request.form['owner']])[0]

        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        mileage = float(request.form['mileage'])
        engine = int(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = float(request.form['seats'])

     
        # Prepare input array
        features = np.array([[name, year, km_driven, fuel, seller_type, transmission,
                              owner, mileage, engine, max_power, seats]])

        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=f"Predicted Price: ₹{int(prediction):,}")

    except Exception as e:
        
        return render_template('index.html', prediction=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=False,port=5050)
