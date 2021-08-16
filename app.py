from flask import Flask, request, redirect, url_for, flash, jsonify, render_template

import joblib
import pandas as pd #dataframe
import numpy as np #mathematical computations

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")



@app.route('/api/', methods=['POST'])
def get_price():
    brand = request.form.get('brand')
    model_year = int(request.form.get('model_year'))
    body_type = request.form.get('body_type')
    transmission = request.form.get('transmission')
    fuel_type = request.form.get('fuel_type')
    engine_capacity = int(request.form.get('engine_capacity'))
    kilometers_run = int(request.form.get('kilometers_run'))
    brand_dict = {'Chevrolet': 0, 'Ford': 1, 'Haval': 2, 'Honda': 3, 'Hyundai': 4, 'Kia': 5, 'Mahindra': 6,
                  'Maruti Suzuki': 7, 'Mazda': 8, 'Mitsubishi': 9, 'Nissan': 10, 'Proton': 11, 'SsangYong': 12,
                  'Suzuki': 13, 'Tata': 14, 'Toyota': 15}
    body_type_dict = {'Estate': 0, 'Hatchback': 1, 'MPV': 2, 'SUV / 4x4': 3, 'Saloon': 4}

    # deciding fuel type
    cng_and_oil = 0
    hybrid = 0
    lpg_and_oil = 0
    oil = 0
    if fuel_type == 'CNG and OIL':
        cng_and_oil = 1
    elif fuel_type == 'HYBRID':
        hybrid = 1
    elif fuel_type == 'LPG and OIL':
        lpg_and_oil = 1
    elif fuel_type == 'OIL':
        oil = 1

    # deciding transmission type
    automatic = 0
    manual = 0
    if transmission == 'Automatic':
        automatic = 1
        manual = 0
    elif transmission == 'Manual':
        automatic = 0
        manual = 1

    test_array = [automatic, manual, cng_and_oil, hybrid, lpg_and_oil, oil, brand_dict[brand], model_year,
                  body_type_dict[body_type], engine_capacity, kilometers_run]
    test_array = np.array(test_array)  # convert into numpy array

    test_array = test_array.reshape(1, -1)  # reshape
    test_df = pd.DataFrame(test_array, columns=['Automatic', 'Manual', 'CNG and OIL', 'HYBRID', 'LPG and OIL', 'OIL',
                                                'brand', 'model_year', 'body_type', 'engine_capacity',
                                                'kilometers_run'])

    # declare path where you saved your model
    model_path = 'xg_boost_model.pkl'
    # open file
    file = open(model_path, "rb")


    # load the trained model
    trained_model = joblib.load(file)
    prediction = int(trained_model.predict(test_df))

    return render_template("prediction.html", prediction = prediction)

if __name__ == '__main__':
     app.run(debug=True)














