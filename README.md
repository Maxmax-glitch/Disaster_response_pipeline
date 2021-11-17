# Disaster_response_pipeline
ETL and NLP on social media messages to classify in specific first responder categories

## Project Overview
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages into different categories relevant for certain first-responder organizations.
Messages are cleaned and transformed according to NLP standard practice. Classification is done within a ML Pipeline with RandomForestClassifier and GridSearch Optimization.
The data is then used in a flask webapp.


## Contents
Raw data in CSV files: 
  disaster_categories.csv - response variables
  disaster_messages.csv - messages to 

process_data.py contains ETL pipeline
train_classifier.py contains ML pipeline

run.py contains the webapp

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Sources & References
 - Project idea and data source from: udacity.com data science degree
 - Webapp, python code structure and main() functions were supplied by udacity.com
 - All other functions were programmed by myself
 - Webapp was expanded by myself with additional visualizations
