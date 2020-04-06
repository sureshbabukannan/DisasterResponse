import sys
import pandas as pd
import numpy as np
import sqlalchemy as sql
from sqlalchemy import inspect
from sqlalchemy.ext.declarative import declarative_base

def load_data(messages_filepath, categories_filepath):
    '''
	INPUT - messages csv file name and file path
		categories csv file name and file path

	OUTPUT - pandas dataframe 

	       1. read the message file into a pandas dataframe
	       2. read the categories file into a pandas dataframe
	       3. merge the messages dataframe and catergories dataframe
	       4. return the merged dataframe
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    df = messages.merge(categories, on=['id'])

    return df


def clean_data(df):
    '''
	INPUT - pandas  dataframe

	OUTPUT - pandas dataframe with cleaned data

	1. create categories dataframe by spliting the categories column by ';'
	2. rename the new columns created by splitting the categories with the category values.
	3. Convert category values to just numbers 0 or 1.
	4. merge the input dataframe and the message categories split columnn
	5. remove any duplicate messages
    '''

    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(";", expand=True)

    # select the first row of the categories dataframe and split the values of the rows with '-'
    row = categories.iloc[0,:].str.split('-').tolist()
 
    # Extract the catergory values from the row and rename the columns
    category_colnames = np.array(row)[:,0].astype(str)
    categories.columns = category_colnames

    #Convert category values to just numbers 0 or 1
    for column in categories:
    	# set each value to be the last character of the string
    	categories[column] = categories[column].str[-1]

    # recast the string to number
    categories[column] = categories[column].astype(int) 
    
    #Replace `categories` column in `df` with new category columns
    df = df.drop(['categories'],axis='columns')
    df = pd.concat([df,categories],axis=1)
     
    # Remove any duplicte messages
    df = df.drop_duplicates(['message'])

    return df


def save_data(df, database_filename):
    '''
        INPUT - panda dataframe , database file name 

	OUTPUT - pandas dataframe data stored to the database file 

	1. Create a SQLlite database using SQLAlchamey packages
	2. Load the input dataframe to the the SQL database.

    '''
    # Create database engine object
    db_url = 'sqlite:///{}'.format(database_filename)
    engine = sql.create_engine(db_url)
       
    #Load input dataframe to database with table name same as the database filename
    table_name = database_filename.split('.')[0]
  
    ins = inspect(engine)

    if table_name in ins.get_table_names():
        try:
            engine.execute("DROP table {}".format(table_name))
        except e:
            print("Error", e)

    df.to_sql(table_name, engine, index=False)

    return db_url
   
def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        db_url = save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')

        engine = sql.create_engine(db_url)

        print('Checking stored data !!')
        result = engine.execute("SELECT * from {} LIMIT 2".format(database_filepath.split(".")[0]))
        
        for r in result: 
            print(r)    

        print('End of data processing')
   
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
