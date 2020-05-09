####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Import dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle

# Functions
from ml_models_functions import getDataForModel
from ml_models_functions import cleanDiabetesData
from ml_models_functions import runLogisticRegression
from ml_models_functions import runGradientBoostingClassifier
from ml_models_functions import runKNeighborsClassifier
from ml_models_functions import runDecisionTreeClassifier

####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Main Script:
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################



### Initialize:
# Intialize error = No
error_flag = False



### Fetch input data for model:
print("*** Get data for analysis ***")
# Intialize 
diabetes_df = pd.DataFrame
# Call function
diabetes_data = getDataForModel()
# Output
error_flag = diabetes_data[0]
diabetes_df = diabetes_data[1]
# Exit if error 
if error_flag == True:
    exit()
# Print for logging
print(("Number of records retrieved: {}").format(diabetes_df['outcome'].count())) 
print("1st 5 rows are: ")
print(diabetes_df.head())



### Clean the input data 
print("*** Perform data cleanup ***")
# Intialize 
clean_diabetes_df = pd.DataFrame
num_rows_removed = 0
# Call function
clean_diabetes_data = cleanDiabetesData(diabetes_df)
# Output
error_flag = clean_diabetes_data[0]
num_rows_removed = clean_diabetes_data[1]
clean_diabetes_df = clean_diabetes_data[2]
# Exit if error 
if error_flag == True:
    exit()
# Print for logging
print(("Number of records removed: {}").format(num_rows_removed)) 
print("1st 5 rows are: ")
print(clean_diabetes_df.head())
print(("Shape of input data: Number of rows: {}  x  Number of features: {}").format(clean_diabetes_df['outcome'].count(), len(clean_diabetes_df.columns)-1))



### Machine learning:  
 

# Establish X & y for Logistic Regression Model 
print("*** Split training/test data ***")
# X =  All data in dataframe except outcome 
X = clean_diabetes_df.loc[:, clean_diabetes_df.columns != 'outcome']
y = clean_diabetes_df['outcome']
# Split data into train & test 
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, stratify=y)
# Print for logging
print("Training data for model:")
print(("Shape of model training data (features): Number of rows: {}  x  Number of features: {}").format(X_train['age'].count(), len(X_train.columns)))
print(("Shape of model training data (results): Number of rows: {}").format(len(y_train)))



# Model 1. Logistic Regression
print("*** Run Logistic Regression ***")
error_flag = runLogisticRegression(X_train, X_test, y_train, y_test)
# Exit if error 
if error_flag == True:
    exit()



# Model 2. Gradient Boosting Classifier
print("*** Run Gradient Boosting Classifier ***")
error_flag = runGradientBoostingClassifier(X_train, X_test, y_train, y_test)
# Exit if error 
if error_flag == True:
    exit()



# Model 3. K Neighbors Classifier
print("*** Run K Neighbors Classifier ***")
error_flag = runKNeighborsClassifier(X_train, X_test, y_train, y_test)
# Exit if error 
if error_flag == True:
    exit()



# Model 3. K Neighbors Classifier
print("*** Run Decision Tree Classifier ***")
error_flag = runDecisionTreeClassifier(X_train, X_test, y_train, y_test)
# Exit if error 
if error_flag == True:
    exit()

####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################



