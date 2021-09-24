from flask import Flask, request, redirect, url_for, flash, jsonify, render_template
import joblib
import pandas as pd #dataframe
import numpy as np #mathematical computations
from sklearn.preprocessing import MinMaxScaler # Min Max Scaler

app = Flask(__name__, template_folder='template')


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")



@app.route('/api/', methods=['POST'])
def get_price():
    car_model = request.form.get('car_model')
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
    car_model_dict = {'320i': 0, '5 Series': 1, 'APV': 2, 'Accent': 3, 'Allion': 4, 'Alphard': 5, 'Alto': 6,
                      'Alto 800': 7, 'Aqua': 8, 'Attrage': 9, 'Auris': 10, 'Avanza': 11, 'Axela': 12, 'Axio': 13,
                      'Bluebird': 14, 'C-Class': 15, 'C-HR': 16, 'CR-V': 17, 'CR-Z': 18, 'CX-7': 19, 'Cami': 20,
                      'Camry': 21, 'Carina': 22, 'Carryboy': 23, 'Cefiro': 24, 'City': 25, 'CityRover': 26, 'Civic': 27,
                      'Coaster': 28, 'Corolla': 29, 'Corona': 30, 'Corsa': 31, 'Crown': 32, 'Discovery': 33,
                      'Dualis': 34, 'Dyna': 35, 'E 250': 36, 'Eco Sport': 37, 'Esquire': 38, 'Estima': 39,
                      'Fielder': 40, 'Fiesta': 41, 'Fit': 42, 'GLA-Class': 43, 'GLX': 44, 'Grace': 45, 'H-RV': 46,
                      'H1': 47, 'H2': 48, 'HR-V': 49, 'Harrier': 50, 'Hiace': 51, 'Hilux': 52, 'Ikon': 53,
                      'Indigo Ecs': 54, 'Insight': 55, 'Juke': 56, 'Kluger': 57, 'Kyron': 58, 'Lancer': 59,
                      'Land Cruiser': 60, 'LiteAce': 61, 'MPV': 62, 'MR2': 63, 'Mark II': 64, 'Murano': 65, 'NX': 66,
                      'Noah': 67, 'Note': 68, 'Other Model': 69, 'Outlandar': 70, 'Outlander': 71, 'Pajero': 72,
                      'Passo': 73, 'Pathfinder': 74, 'Prado': 75, 'Premio': 76, 'Prius': 77, 'Probox': 78, 'Q5': 79,
                      'RAV4': 80, 'RVR': 81, 'RX': 82, 'RX-8': 83, 'Ractis': 84, 'Raum': 85, 'RunX': 86, 'Rush': 87,
                      'S660': 88, 'Santa Fe': 89, 'Satria': 90, 'Sienta': 91, 'Sonata': 92, 'Spacio': 93, 'Spark': 94,
                      'Sportage': 95, 'Sprinter': 96, 'Starlet': 97, 'Starlet Soleil': 98, 'Succeed': 99, 'Sunny': 100,
                      'Swift': 101, 'Terrano': 102, 'Tiida': 103, 'TownAce': 104, 'Tucson': 105, 'Urvan': 106,
                      'V6': 107, 'Vezel': 108, 'Vista': 109, 'Vitz': 110, 'WagonR': 111, 'Wish': 112, 'X Assista': 113,
                      'X-Trail': 114, 'XJ': 115, 'Yaris': 116, 'ist': 117, 'l200': 118, 'l300': 119}
    features = ['Automatic', 'CNG and OIL', 'HYBRID', 'LPG and OIL', 'OIL', 'brand', 'car_model', 'model_year',
                'body_type', 'engine_capacity', 'kilometers_run']
    scale_vars = ['Automatic', 'CNG and OIL', 'HYBRID', 'LPG and OIL', 'OIL', 'body_type', 'engine_capacity',
                  'kilometers_run']


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


    test_array = [automatic, cng_and_oil, hybrid, lpg_and_oil, oil, brand_dict[brand], car_model_dict[car_model], model_year, body_type_dict[body_type], engine_capacity, kilometers_run]
    test_array = np.array(test_array)  # convert into numpy array

    test_array = test_array.reshape(1, -1)  # reshape
    test_df = pd.DataFrame(test_array, columns=features)

    # scaling data
    scalar_filename = "min_max_scaler.save"
    scalar = joblib.load(scalar_filename)
    scalar.clip = False
    test_df[scale_vars] = scalar.transform(test_df[scale_vars])


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














