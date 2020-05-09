####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Import dependencies
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import pickle


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Get data for model
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def getDataForModel() -> []:

    # Intialize error flag
    error_flag = False    
    # Intialize output dataframne
    diabetes_df = pd.DataFrame

    try:
        # Fetch the data 
        # SMTODO - Change to call flask API to get data
        diabetes_df = pd.read_csv('https://finalproject-jssicc.s3.us-east-2.amazonaws.com/diabetes.csv')
        # Temporary code:
        # Rename coulumns 
        diabetes_df = diabetes_df.rename(columns={'Pregnancies': 'pregnancies', 'Glucose': 'glucose', \
                                                    'BloodPressure': 'blood_pressure', 'SkinThickness': 'skin_thickness', \
                                                    'Insulin': 'insulin', 'BMI': 'bmi', \
                                                    'DiabetesPedigreeFunction':'diabetes_pedigree_function', \
                                                    'Age': 'age', 'Outcome':'outcome'})
    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag, diabetes_df


####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Clean inout diabetes data 
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def cleanDiabetesData(diabetes_df) -> []:

    # Intialize error flag
    error_flag = False    
    # Intialize output dataframne & number of rows removed
    clean_diabetes_df = pd.DataFrame
    num_records_removed = 0


    try:
        # Copy input data to output dataframne
        clean_diabetes_df = diabetes_df.copy()
        # Number of records prior to cleanup
        num_records_orig = clean_diabetes_df['outcome'].count()

        # Remove any rows will all nulls
        clean_diabetes_df = clean_diabetes_df.dropna()

        # Remove rows with zero glucose 
        clean_diabetes_df = clean_diabetes_df[(clean_diabetes_df['glucose'] != 0)]

        # Remove rows with zero blood pressure 
        clean_diabetes_df = clean_diabetes_df[(clean_diabetes_df['blood_pressure'] != 0)]

        # Remove rows with zero blood bmi
        clean_diabetes_df = clean_diabetes_df[(clean_diabetes_df['bmi'] != 0)]

        #### Note:
        #### There are a number or records with zero value for Insulin & Skin thickness
        #### If we remove those records too - not much data is left for analysis 
        #### For the purpose of this analysis - records with zero for Insulin & Skin thickness are not removed 


        # Number of records after cleanup
        num_records_clean = clean_diabetes_df['outcome'].count()

        # Number of records removed
        num_records_removed = num_records_orig - num_records_clean


    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag, num_records_removed, clean_diabetes_df

    
####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Run Logistci Regression Model  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def runLogisticRegression(X_train, X_test, y_train, y_test) -> bool:


    # Intialize error flag
    error_flag = False    
    

    try:
        # Create a Logistic Regression Model
        diabetes_model = LogisticRegression(solver='lbfgs', max_iter=200)

        # Fit the model on training data 
        diabetes_model.fit(X_train, y_train)

        # Save the model 
        filename = "FlaskWebProject/models/LogisticRegression/DiabetesLogisticRegressionModel.sav"
        pickle.dump(diabetes_model, open(filename, 'wb'))

        # Model score:
        model_score_training = diabetes_model.score(X_train, y_train)
        print(("Model Score (Training data): {}").format(model_score_training))
        model_score_test = diabetes_model.score(X_test, y_test)
        print(("Model Score (Test data): {}").format(model_score_test))
        compare_scores_df = pd.DataFrame({"Model": ["Logistic Regression"], "Model Score (Training Data)": [model_score_training], "Model Score (Test Data)": [model_score_test]})
        compare_scores_df.to_csv('FlaskWebProject/models/LogisticRegression/output/LogisticRegressionScores.csv', index=False)

        # Comparison of predicted results vs. actual on test data:
        # Actual results
        actual_results = y_test
        # Predicted results 
        predicted_results = diabetes_model.predict(X_test)
        compare_results_df = pd.DataFrame({"Predicted Results": predicted_results, "Actual Results": actual_results})
        compare_results_df = pd.concat([X_test,compare_results_df], axis=1)
        compare_results_df.to_csv('FlaskWebProject/models/LogisticRegression/output/LogisticRegressionResults.csv', index=False)

        # Print for logging
        print("Logistic regression completed, model and results saved")


    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag

    
####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Run Gradient Boosting Classifier Model  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def runGradientBoostingClassifier(X_train, X_test, y_train, y_test) -> bool:


    # Intialize error flag
    error_flag = False    
    

    try:
        # Create a Gradient Boosting Classifier Model
        diabetes_model = GradientBoostingClassifier()

        # Fit the model on training data 
        diabetes_model.fit(X_train, y_train)

        # Save the model 
        filename = "FlaskWebProject/models/GradientBoostingClassifier/DiabetesGradientBoostingClassifierModel.sav"
        pickle.dump(diabetes_model, open(filename, 'wb'))

        # Model score:
        model_score_training = diabetes_model.score(X_train, y_train)
        print(("Model Score (Training data): {}").format(model_score_training))
        model_score_test = diabetes_model.score(X_test, y_test)
        print(("Model Score (Test data): {}").format(model_score_test))
        compare_scores_df = pd.DataFrame({"Model": ["Gradient Boosting Classifier"], "Model Score (Training Data)": [model_score_training], "Model Score (Test Data)": [model_score_test]})
        compare_scores_df.to_csv('FlaskWebProject/models/GradientBoostingClassifier/output/GradientBoostingClassifierScores.csv', index=False)

        # Comparison of predicted results vs. actual on test data:
        # Actual results
        actual_results = y_test
        # Predicted results 
        predicted_results = diabetes_model.predict(X_test)
        compare_results_df = pd.DataFrame({"Predicted Results": predicted_results, "Actual Results": actual_results})
        compare_results_df = pd.concat([X_test,compare_results_df], axis=1)
        compare_results_df.to_csv('FlaskWebProject/models/GradientBoostingClassifier/output/GradientBoostingClassifierResults.csv', index=False)

        # Print for logging
        print("Gradient Boosting Classifier completed, model and results saved")


    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag

    
####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Run KNeighborsClassifier Model  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def runKNeighborsClassifier(X_train, X_test, y_train, y_test) -> bool:


    # Intialize error flag
    error_flag = False    
    

    try:
        # Create a K Neighbors Classifier Model
        diabetes_model = KNeighborsClassifier()

        # Fit the model on training data 
        diabetes_model.fit(X_train, y_train)

        # Save the model 
        filename = "FlaskWebProject/models/KNeighborsClassifier/DiabetesKNeighborsClassifierModel.sav"
        pickle.dump(diabetes_model, open(filename, 'wb'))

        # Model score:
        model_score_training = diabetes_model.score(X_train, y_train)
        print(("Model Score (Training data): {}").format(model_score_training))
        model_score_test = diabetes_model.score(X_test, y_test)
        print(("Model Score (Test data): {}").format(model_score_test))
        compare_scores_df = pd.DataFrame({"Model": ["K Neighbors Classifier"], "Model Score (Training Data)": [model_score_training], "Model Score (Test Data)": [model_score_test]})
        compare_scores_df.to_csv('FlaskWebProject/models/KNeighborsClassifier/output/KNeighborsClassifierScores.csv', index=False)

        # Comparison of predicted results vs. actual on test data:
        # Actual results
        actual_results = y_test
        # Predicted results 
        predicted_results = diabetes_model.predict(X_test)
        compare_results_df = pd.DataFrame({"Predicted Results": predicted_results, "Actual Results": actual_results})
        compare_results_df = pd.concat([X_test,compare_results_df], axis=1)
        compare_results_df.to_csv('FlaskWebProject/models/KNeighborsClassifier/output/KNeighborsClassifierResults.csv', index=False)

        # Print for logging
        print("K Neighbors Classifier completed, model and results saved")


    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag

    
####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
# Function: Run Decision Tree Classifier Model  
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################
def runDecisionTreeClassifier(X_train, X_test, y_train, y_test) -> bool:


    # Intialize error flag
    error_flag = False    
    

    try:
        # Create a Decision Tree Classifier Model
        diabetes_model = DecisionTreeClassifier()

        # Fit the model on training data 
        diabetes_model.fit(X_train, y_train)

        # Save the model 
        filename = "FlaskWebProject/models/DecisionTreeClassifier/DiabetesDecisionTreeClassifierModel.sav"
        pickle.dump(diabetes_model, open(filename, 'wb'))

        # Model score:
        model_score_training = diabetes_model.score(X_train, y_train)
        print(("Model Score (Training data): {}").format(model_score_training))
        model_score_test = diabetes_model.score(X_test, y_test)
        print(("Model Score (Test data): {}").format(model_score_test))
        compare_scores_df = pd.DataFrame({"Model": ["Decision Tree Classifier"], "Model Score (Training Data)": [model_score_training], "Model Score (Test Data)": [model_score_test]})
        compare_scores_df.to_csv('FlaskWebProject/models/DecisionTreeClassifier/output/DecisionTreeClassifierScores.csv', index=False)

        # Comparison of predicted results vs. actual on test data:
        # Actual results
        actual_results = y_test
        # Predicted results 
        predicted_results = diabetes_model.predict(X_test)
        compare_results_df = pd.DataFrame({"Predicted Results": predicted_results, "Actual Results": actual_results})
        compare_results_df = pd.concat([X_test,compare_results_df], axis=1)
        compare_results_df.to_csv('FlaskWebProject/models/DecisionTreeClassifier/output/DecisionTreeClassifierResults.csv', index=False)

        # Print for logging
        print("Decision Tree Classifier completed, model and results saved")


    except Exception as e:
        print(('Exception occured : {}').format(e))
        error_flag = True


    # Return error flag & data dataframne 
    return error_flag

    
####################################################################################################################################################################
# ---------------------------------------------------------------------------------------------------------------------------------------------------------------- #
####################################################################################################################################################################



