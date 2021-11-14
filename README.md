# Disaster_response_pipeline
ETL and NLP on social media messages to classify in specific first responder categories

Source: udacity.com data science degree


## Project Overview
Analyze disaster data from Figure Eight to build a model for an API that classifies disaster messages into different categories relevant for certain first-responder organizations.
Messages are cleaned and transformed according to NLP standard practice. Classification is done within a ML Pipeline with RandomForestClassifier and GridSearch Optimization.

## Contents
Raw data in CSV files: 
  disaster_categories.csv - response variables
  disaster_messages.csv - messages to 

process_data.py contains ETL pipeline
train_classifier.py contains ML pipeline

