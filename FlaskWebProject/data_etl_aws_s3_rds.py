#!/usr/bin/env python
# coding: utf-8

# In[5]:


# Dependencies
import pandas as pd
from sqlalchemy import create_engine

# Read csv file from cloud storage amazon aws s3 into dataframe
data_file = "https://finalproject-jssicc.s3.us-east-2.amazonaws.com/diabetes.csv"
data_df = pd.read_csv(data_file)

# Rename column headers 
data_df = data_df.rename(columns={"Pregnancies": "pregnancies", 
                                  "Glucose": "glucose",
                                  "BloodPressure": "blood_pressure",
                                  "SkinThickness": "skin_thickness",
                                  "Insulin": "insulin",
                                  "BMI": "bmi",
                                  "DiabetesPedigreeFunction": "diabetes_pedigree_function",
                                  "Age": "age",
                                  "Outcome": "outcome"})

# Clean data by dropping duplicates and null values
data_df.drop_duplicates(inplace=True)
data_df.dropna(inplace=True)

# Get database user and password 
from config import dbuser, dbpassword
rds_connection_string = f"{dbuser}:{dbpassword}@finalproject-jssicc.cjfhkorsupva.us-east-2.rds.amazonaws.com:5432/project_db"
engine = create_engine(f'postgresql://{rds_connection_string}')

# Check for tables
# engine.table_names()

# Convert DataFrame to SQL and insert into the table 'diabetes_info'
# Drop table if it exists and create a new one with the data 
data_df.to_sql(name='diabetes_info', con=engine, if_exists='replace', index=False)

# Confirm data has been added by querying the diabetes_info table
pd.read_sql_query('select * from diabetes_info', con=engine).head()
pd.read_sql_query('select count(*) from diabetes_info', con=engine)


# In[ ]:




