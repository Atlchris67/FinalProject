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

def savePredictedResults(diabetes_results_df):
    print("*** Saving the result predicted getapiinfo ***")
    engine = db.engine
    conn = engine.connect()
    diabetes_results_df.to_sql(name = "diabetes_results", con = conn, if_exists = "append", index = False, schema = "public")
    return 

# prediction function 
def ValuePredictor(to_predict_list): 
    #predictDiabetics(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age)
    result = [1]
    return result[0] 
