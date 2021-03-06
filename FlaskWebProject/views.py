"""
Routes and views for the flask application.
"""

from datetime import datetime
from flask import render_template
from FlaskWebProject import app
from flask import request
from FlaskWebProject.pgsql import ValuePredictor, getDBData

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
        year=datetime.now().year,
    )

@app.route('/methodsGB')
def methodsGB():
    """Renders the about page."""
    return render_template(
        'methodsGB.html',
        year=datetime.now().year,
    )

@app.route('/methodsKN')
def methodsKN():
    """Renders the about page."""
    return render_template(
        'methodsKN.html',
        year=datetime.now().year,
    )

@app.route('/methodsLR')
def methodsLR():
    """Renders the about page."""
    return render_template(
        'methodsLR.html',
        year=datetime.now().year,
    )

@app.route('/othermethods')
def othermethods():
    """Renders the about page."""
    return render_template(
        'methodsDT.html',
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
    """Renders the prediction page."""
    return render_template(
        'predict.html',
        year=datetime.now().year,
    )

    
@app.route('/results', methods = ['POST']) 
def results(): 
    """Renders the result  page."""
     
    to_predict_list = request.form.to_dict() 
    to_predict_list = list(to_predict_list.values()) 
    to_predict_list = list(map(float, to_predict_list)) 
    result = ValuePredictor(to_predict_list)         
    if int(result)== 1: 
        prediction ='Run Martha, or you\'re gonna get the sugar.'
    else: 
        prediction ='Go ahead and have another donut Martha, you\'re all good.'            
    return render_template("results.html",
            year=datetime.now().year,
             prediction = prediction
             ) 

@app.route('/dataset')
def dataset(value=None):
    """Renders the dataset page."""
    data = getDBData()
    return render_template("dataset.html",
        value=data
    )