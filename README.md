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

- README.md: read me file
- \app
	- run.py: Flask file that runs the web app
   	- \templates
		- master.html: main page of the web application 
		- go.html: result web page
- \data
	- disaster_categories.csv: dataset containing the categories to which each message belongs
	- disaster_messages.csv: dataset contaning the messages sent during disaster events
	- DisasterResponse.db: the exported Sqlite database containing the cleaned data
	- process_data.py: Python file that runs the ETL pipeline and exports a SQLite database
	
- \model
	- train_classifier.py: Python file that runs the ML pipeline and exports the model as a pickle file
	- classifier.pkl: the saved model in a pickle file


## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

## ETL pipeline description

## Machine Learning pipeline description

## Results

## Licensing, Authors and Other Acknowledgements
I must give credit to Bilal Yussef who puplished the data on Kaggle [here](https://www.kaggle.com/bilalyussef/google-books-dataset). All the licensing for the data and other information will be available at that link.





