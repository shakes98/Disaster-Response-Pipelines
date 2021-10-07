#Import the libraries
import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''The load_data function will load in the data from the messages and categories dataframes.
    
    This function will merge the datasets using common id and then return a pandas dataframe. 
    
    INPUT:
    messages_filepath (str):  File path of messages data
    categories_filepath (str): File path of categories data
    
    OUTPUT:
    df: A pandas data frame containing the merged data
    '''
    
    #load the messages dataset
    messages = pd.read_csv(messages_filepath)
     
    #load the categories dataset
    categories = pd.read_csv(categories_filepath)
    
    #merge the two datasets
    df = pd.merge(messages, categories, on='id', how = 'left')
    
    return df



def clean_data(df):
    ''' This clean_data function will go through and clean the inputted dataframe.
    
    The function will:
        1. Split the categories into separate category columns
        2. Rename the columns of 'categories'
        3. Convert category values to just numbers 0 or 1 (booleans)
        4. Replace categories column in df with new category columns
        5. Drop rows in replaced column that have 2 as a value
        6. Remove duplicates

    INPUT:
    df:  A pandas dataframe
    
    OUTPUT:
    df: A cleaned pandas dataframe which has category columns and boolean responses
    '''
    #Split the dataframes category column into individual category columns
    categories = df["categories"].str.split(";", expand=True)
        
    #Rename every column to its corresponding category
    
    #Select the first rows of the categories dataframe:
    row = categories.iloc[0,:]
    
    #Use this row to extract a list of new column names for categories
    #This uses a lambda function that takes everything up to the second
    #to last character of each tring with slicing
    cat_colnames = row.apply(lambda x: x[:-2])
        
    #rename the column names:
    categories.columns = cat_colnames
   
    
    #Convert category values to just numbers 0 and 1 (boolean)
    
    #Iterate through the category columns in df to keep only the last character of each string (the 1 or 0) 
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split("-").str[-1]

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
    #replaces 2's with 1's:
    categories= categories.replace([2], 1)
        

        
    #Replace categories column in the df with the new category columns
    
    #Drop categories column from df
    df = df.drop(['categories'], axis=1)   
    
    #Concatenate the original datafram with the new categories dataframe
    df = pd.concat([df, categories], axis=1)

    
    #Drop any duplicates
    df = df.drop_duplicates()
    
    return df
    

def save_data(df, database_filename):
    ''' The save_data function will save dataframe into a SQLite database file. 
    
    INPUT:
    df: A pandas dataframe
    database_filename (str): The name of the database you want to save it as
    
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Disaster_Response', engine, index=False, if_exists='replace')    
    
    

def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
