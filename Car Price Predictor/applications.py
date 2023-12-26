from flask import Flask, render_template, request
import pandas as pd 
import pickle
import numpy as np



app = Flask(__name__)

model = pickle.load(open('Car_Price_Predictor.pkl','rb'))

# importing the csv file to fetch categories 

car = pd.read_csv("cleaned_car.csv")

@app.route('/')
def index():
    companies = sorted(car["company"].unique())
    car_models = sorted(car["name"].unique())
    year = sorted(car["year"].unique(), reverse=True)
    fuel_type = sorted(car["fuel_type"].unique())
    #companies.insert(0, "Select Company")
    return render_template('index.html', companies=companies, car_models=car_models, year=year, fuel_type=fuel_type)


@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company_id')
    car_model = request.form.get('car_model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel')
    kilo = int(request.form.get('kilo'))

    print(company, car_model, year, fuel_type, kilo)
    prediction = model.predict(pd.DataFrame([[car_model, company, year, kilo, fuel_type]], columns=['name', 'company', 'year', 'kms_driven', 'fuel_type']))
    print("predicted value is ", prediction)
    return str(np.round(prediction[0], 2))

if __name__ == '__main__':
    app.run(debug=True)