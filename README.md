# Disaster Response Pipeline Project
This is a repository that will contain all my files for the Udacity Data Science Nanodegree submission - Disaster Response Pipelines Project. This will contain ETL and ML pipelines


## Table of Contents
1. [Installations Needed](#installations-needed)
2. [Project Motivation](#project-motivation)
3. [File Descriptions](#file-descriptions)
4. [Instructions](#instructions)
5. [ETL Pipeline Description](#etl-pipeline-description)
6. [Machine Learning Pipeline Description](#machine-learning-pipeline-description)
7. [Results & Learnings](#results-&-learnings)
8. [Licencing, Authors and Other Acknowledgements](#licensing-authors-and-other-acknowledgements)

## Installations Needed
Beyond the installation of the Anaconda distribution of python, there should be no other necessary libraries needed to run the code. All libraries required will be installed within the python code. There should be no issues running the code when using Python Version 3. The data needed is saved in the github repository along with the code. To write the code, I used a jupyter noteboook and then python files.

## Project Motivation
For this project, I am going to use software engineering skills and data engineering skills, taught in the Udacity Data Science nanodegree, to analyze disaster data from Figure Eight. This data set contains real messages sent during disasters. 

Specifically, in this project I will clean this data (using an ETL pipeline) then create a machine learning pipeline to categorize these messages into 36 different disaster categories, so that these messages could be sent to the appropriate disaster response agency. Using this, I will create a web app for an emergency worker to input a message and then get the classification of the message.

## File Descriptions

- README.md: read me file
- app
	- run.py: Flask file that runs the web app
   	- templates
		- master.html: main page of the web application 
		- go.html: result web page
- data
	- disaster_categories.csv: dataset containing the categories of messages
	- disaster_messages.csv: dataset contaning the messages sent 
	- DisasterResponse.db: the exported database uploaded in SQLite. This is the cleaned data
	- process_data.py: Python file that runs the ETL pipeline. This is the file that exports the SQLite database above.
	
- model
	- train_classifier.py: Python file that runs the ML pipeline. This reads from the SQLite database and will export the file as a pickle
	- classifier.pkl: the saved model in a pickle file. Running the code will create a pickle file however this file itself was too large to be uploaded to github (https://knowledge.udacity.com/questions/547777)


## Instructions
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app (should be /home/workspace/app and can get to this by doing cd app in the terminal).
    `python run.py`

3. Go to http://0.0.0.0:3001/

## ETL pipeline description
1. Loads the message.csv and the categories.csv data sets
2. Merges these two data sets using ID
3. Splits the categories column into seperate columns
4. Converts these to binary values
5. Drops duplicates
6. Stores this clean data in a SQLite database in a specified file path

## Machine Learning pipeline description
1. Loads the cleaned data from the SQLite database from specified path
2. Uses the tokenize function to break the text into tokens, replace URLs and lematizes the tokens
3. Build a model using a pipeline object. This uses CounterVectorizer, TfidfTransformer and MultiOutputClassifier with RandomForestClassifier
4. Trains the model using only training data
5. Evaluates model on test data
6. Saves model as a pickle file to be used in the web app

In this dataset some labels like water have very few examples (so the dataset is imbalanced). Classifiers tend not to perform that well on unbalanced datasets as they can potentially classify the main class well but at the expense of the smaller classes.  For this imbalance, we want to improve the performance of single classifier, so we use the Random Forest Classifieer. MultiOutputClassifier is then used as there is multiple categorical columns. In future it may be worth using Balanced Random Forest Classifier instead to counteract the inbalance some more to improve the model.

## Results & Learnings
The result of this is a web app, which can be accessed using the instructions above.

## Licensing, Authors and Other Acknowledgements
I'd would like to thank Figure Eight for providing a great data set to practice machine learning and NLP techniques on, in this classification project.

I'd also like to thank Udacity and the instructors for the courses on software engineering and data engineering.



