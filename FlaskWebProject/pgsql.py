import numpy as np
import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, func

from flask import Flask, jsonify
from flask_sqlalchemy import SQLAlchemy

import pandas as pd
from FlaskWebProject import app
import os

from sklearn.linear_model import LogisticRegression
import pickle
from datetime import date

# https://finalproject-jssicc.s3.us-east-2.amazonaws.com/diabetes.csv
DATABASE_URL = os.environ.get('DATABASE_URL', '') or "postgres://postgres:finalproject-jssicc@finalproject-jssicc.cjfhkorsupva.us-east-2.rds.amazonaws.com:5432/project_db"

app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
db = SQLAlchemy(app)


def getApiInfo():
    print("entering getapiinfo")
    engine = db.engine
    conn = engine.connect()
    data = pd.read_sql("SELECT * FROM apikey where id=1", conn)
    print(data)
    key= data['api_key'][0]
    baseurl = data['base_url'][0]
    print(key)
    return key , baseurl

def getDBData():
    print("entering getDBData")
    engine = db.engine
    conn = engine.connect()
    data_df = pd.read_sql("SELECT * FROM diabetes_info", conn)
    json_data = data_df.to_json(orient='table')
    # save the table data as json file
    data_df.to_json("FlaskWebProject/data/diabetes.json")
    return json_data

def savePredictedResults(diabetes_results_df):
    print("*** Saving the result predicted getapiinfo ***")
    engine = db.engine
    conn = engine.connect()
    diabetes_results_df.to_sql(name = "diabetes_results", con = conn, if_exists = "append", index = False, schema = "public")
    return 

# prediction function 
def ValuePredictor(to_predict_list): 
    
    error_flag, predicted_result =  predictDiabetes(to_predict_list[0], to_predict_list[1], to_predict_list[2], to_predict_list[3], to_predict_list[4], to_predict_list[5], to_predict_list[6], to_predict_list[7],  "USA", "GA", "female")

    return predicted_result 


def predictDiabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, user_country, user_state, gender) -> []: 

    ### Initialize:
    # Intialize error = No
    error_flag = False
    predicted_result = None

    try:


        
        # Predict result:


        # Convert input to dataframe
        input = {"pregnancies":[pregnancies], "glucose":[glucose], "blood_pressure":[blood_pressure], \
                 "skin_thickness":[skin_thickness], "insulin":[insulin], "bmi":[bmi], \
                 "diabetes_pedigree_function":[diabetes_pedigree_function], "age":[age]}
        input_df = pd.DataFrame(input)

        # Load the model from disk
        filename = "FlaskWebProject/models/LogisticRegression/DiabetesLogisticRegressionModel.sav"
        diabetes_model = pickle.load(open(filename, 'rb'))

        # Predict the result from model 
        predicted_result = diabetes_model.predict(input_df)[0]
        print(("Predicted result : {}").format(predicted_result))



        # Save predicted result to Database
        
        # Curret date
        current_date = date.today()
        other_input = {"create_date":[current_date], "user_country":[user_country], "user_state":[user_state], "gender":[gender]}
        other_input_df = pd.DataFrame(other_input)

        # result
        result = {"outcome":[predicted_result]}
        result_df = pd.DataFrame(result)
        
        # Concatenate the 3 dataframes to match table 
        diabetes_results_df = pd.concat([input_df, result_df, other_input_df], axis=1)
        print(diabetes_results_df.head())

        # Save
        # savePredictedResults(diabetes_results_df)



    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True

    
    # Return
    return error_flag, predicted_result
