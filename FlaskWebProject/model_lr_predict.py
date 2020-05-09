####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Import dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle
from datetime import date

# from pgsql import savePredictedResults


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Main Function : Predict diabetes (Logistic Regression)
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def predictDiabetes(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, user_name, user_country, user_state, gender) -> []: 

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
        result = {"outcome":[predicted_result]}
        result_df = pd.DataFrame(result)
        
        # Curret date
        current_date = date.today()
        other_input = {"create_date":[current_date], "user_name":[user_name], "user_country":[user_country], "user_state":[user_state], "gender":[gender]}
        other_input_df = pd.DataFrame(other_input)

        # result
        

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


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
