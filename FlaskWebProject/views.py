"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject import app
from flask import request
from FlaskWebProject.pgsql import ValuePredictor

@app.route('/')
@app.route('/home')
def home():
    """Renders the home page."""
    return render_template(
        'index.html',
        year=datetime.now().year,
    )

@app.route('/methodsDT')
def methodsDT():
    """Renders the about page."""
    return render_template(
        'methodsDT.html',
        title='Machine Learning Methods Used',
        year=datetime.now().year,
    )

@app.route('/methodsGB')
def methodsGB():
    """Renders the about page."""
    return render_template(
        'methodsGB.html',
        title='Machine Learning Methods Used',
        year=datetime.now().year,
    )

@app.route('/methodsKN')
def methodsKN():
    """Renders the about page."""
    return render_template(
        'methodsKN.html',
        title='Machine Learning Methods Used',
        year=datetime.now().year,
    )

@app.route('/methodsLR')
def methodsLR():
    """Renders the about page."""
    return render_template(
        'methodsLR.html',
        year=datetime.now().year,
    )

@app.route('/contact')
def contact():
    a = 1
    a += 1
    """Renders the contact page."""
    return render_template(
        'contact.html',
        year=datetime.now().year,
    )

@app.route('/predict')
def predict():
    a = 1
    a += 1
    """Renders the prediction page."""
    return render_template(
        'predict.html',
        title='Diabetes Prediction Form',
        year=datetime.now().year,
        message='Are you at risk?'
    )

    
@app.route('/results', methods = ['POST']) 
def results(): 
    a = 1
    a += 1
    """Renders the result  page."""
     
    to_predict_list = request.form.to_dict() 
    to_predict_list = list(to_predict_list.values()) 
    to_predict_list = list(map(float, to_predict_list)) 
    result = ValuePredictor(to_predict_list)         
    if int(result)== 1: 
        prediction ='Run Martha your gonna get the sugar.'
    else: 
        prediction ='Go ahead and have another donut Martha, your all good.'            
    return render_template("results.html",
            title='Diabetes Prediction Form',
            year=datetime.now().year,
            message='Are you at risk?',
             prediction = prediction
             ) 

@app.route('/dataset')
def dataset(value=None):
    """Renders the dataset page."""
    data = getDBData()
    return render_template("dataset.html",
        value=data
    )