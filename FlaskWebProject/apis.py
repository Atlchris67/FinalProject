"""
Routes and views for the api application.
"""

from datetime import datetime
from flask import render_template , jsonify
from FlaskWebProject import app
import json
import requests
from FlaskWebProject.pgsql import getApiInfo, getDBData


@app.route('/table')
def table():
    """Renders the contact page."""
    print("rendering table from python application")
    # get dataset from database
    data = getDBData()
    return data

@app.route('/chart')
def chart():
    """Renders the contact page."""
    print("rendering chart from python application")
    # get api info from database
    apikey, baseurl = getApiInfo()
    queryUrl = baseurl + "&api_key="+ apikey
    response = requests.get(queryUrl).json()
    return response



