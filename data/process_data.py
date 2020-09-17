''' This ETL program will load two files, transform them, and load the prepared version to a database

Parameters:
    argument 1: Messages CSV file
    argument 2: Categories CSV file
    
Returns:
    .db file: a database file containing the cleaned data

'''


import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''load the messages and categories files and merge them on id
    Parametesr:
        messages_filepath
        categories_filepath
    
    Returns:
        df: the dataframe from the merger of the two files
    '''
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    #printing the size of the messages and categoreis dataframe
    print("Messages dataframe shape",messages.shape)
    print("categories dataframe shape",categories.shape)
    
    #merge the two data frames on ids
    df = pd.merge(messages,categories,on='id',how='left')
    return df

def clean_data(df):
    '''Cleans the data and return the cleaned dataframe
    Parameters:
        df: dataframe to clean
    Returns:
        df: cleanded dataframe
    '''
    #split categoreies into seperate columns
    categories = df['categories'].str.split(';',expand=True)
    
    #selecting the first row of the categories dataframe
    first_row = categories.iloc[0,:]
    
    #extract column names
    category_colnames = first_row.apply(lambda x: x[0:-2])
    #assign column names to dataframe
    categories.columns = category_colnames
    #convert category values to 0s and 1s
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = pd.to_numeric(categories[column])
    
    #replace the existing categories column with the splitted version
    df = df.drop('categories',axis=1)
    df = pd.concat([df,categories],axis=1)
    #remove duplicates
    print("complete duplicates",df.duplicated().sum())
    print("id duplicates",df['id'].duplicated().sum())
#     Considering remooving duplicate messages
#     print('messages duplicates',df['message'].duplicated().sum())

    df = df.drop_duplicates()
    df = df.drop_duplicates('id')
#     df = df.drop_duplicates('message')

    print("After removing duplicates")
    print("complete duplicates",df.duplicated().sum())
    print("id duplicates",df['id'].duplicated().sum())
#     Considering remooving duplicate messages
#     print('messages duplicates',df['message'].duplicated().sum())

    return df


def save_data(df, database_filename):
    '''Save the dataframe in the database
    Parameters:
        df: dataframe to save
        database_filename: the name of the database
    '''
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('MessagesTable2',engine,index=False)


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