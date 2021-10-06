# Disaster Response Pipeline Project
This is a repository that will contain all my files for the Udacity Data Science Nanodegree submission - Disaster Response Pipelines Project. This will contain ETL and ML pipelines


## Table of Contents
1. [Installations Needed](#installations-needed)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Results of Analysis](#results-of-analysis)
5. [Licencing, Authors and Other Acknowledgements](#licensing-authors-and-other-acknowledgements)

## Installations Needed
Beyond the installation of the Anaconda distribution of python, there should be no other necessary libraries needed to run the code. All libraries required will be installed within the python code. There should be no issues running the code when using Python Version 3. The data needed is saved in the github repository along with the code. To write the code, I used a jupyter noteboook and then python files.

## Project Motivation
For this project, I am going to use software engineering skills and data engineering skills, taught in the Udacity Data Science nanodegree, to analyze disaster data from Figure Eight. This data set contains real messages sent during disasters. 

Specifically, in this project I will clean this data (using an ETL pipeline) then create a machine learning pipeline to categorize these messages into 36 different disaster categorie, so that these messages could be sent to the appropriate disaster response agency. Using this, I will create a web app for an emergency worker to input a message and then get the classification of the message.

## File Descriptions

app: 
- templates
master.html (main page of the web app)
go.html (classification page of the web app)
run.py (Flask file that runs the web app)
data

disaster_categories.csv (dataset contaning the messages sent during disaster events)
disaster_messages.csv (dataset containing the categories to which each message belongs)
process_data.py (Python file that runs the ETL pipeline and exports a SQLite database)
InsertDatabaseName.db (the exported Sqlite database containing the cleaned data)
models

train_classifier.py (Python file that runs the ML pipeline and exports the model as a pickle file)
classifier.pkl (the saved model)

## Instructions

## ETL pipeline description

## Machine Learning pipeline description

## Results

## Licensing, Authors and Other Acknowledgements
I must give credit to Bilal Yussef who puplished the data on Kaggle [here](https://www.kaggle.com/bilalyussef/google-books-dataset). All the licensing for the data and other information will be available at that link.





