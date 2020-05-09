####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Import dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Main Function : Predict diabetics (Logistic Regression)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def predictDiabetics(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age) -> []: 

    ### Initialize:
    # Intialize error = No
    error_flag = False
    predicted_result = None

    try:

        # Convert input to dataframe
        input = {"pregnancies":[pregnancies], "glucose":[glucose], "blood_pressure":[blood_pressure], \
                 "skin_thickness":[skin_thickness], "insulin":[insulin], "bmi":[bmi], \
                 "diabetes_pedigree_function":[diabetes_pedigree_function], "age":[age]}
        input_df = pd.DataFrame(input)

        # Load the model from disk
        filename = "FlaskWebProject/models/LogisticRegression/DiabeticsLogisticRegressionModel.sav"
        diabetics_model = pickle.load(open(filename, 'rb'))

        # Predict the result from model 
        predicted_result = diabetics_model.predict(input_df)[0]
        print(("Predicted result : {}").format(predicted_result))
        
    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True

    
    # Return
    return error_flag, predicted_result


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
