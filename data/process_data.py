import sys
import pandas as pd
import sqlite3

def load_data(messages_filepath, categories_filepath):
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    return messages,categories


def clean_data(messages,categories):
    
    # merge datasets
    df = messages.merge(categories,how='inner',on='id')
    # create a dataframe of the 36 individual category columns
    categories = categories['categories'].str.split(pat=';', n=-1, expand=True)
    # select the first row of the categories dataframe
    row = categories.loc[0]
    # use this row to extract a list of new column names for categories.
    category_colnames=[]
    for item in range(len(row)):
         category_colnames.append(row[item].split('-')[0])
     # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
    # set each value to be the last character of the string
         categories[column] = categories[column].str.split('-',expand=True)[1]
        # convert column from string to numeric
         categories[column] = categories[column].astype(int)
    # drop the original categories column from `df`
    df.drop('categories',axis=1, inplace=True)
    # drop the original column from `df`
    df.drop('original',axis=1,inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories],axis=1) 
    # drop duplicates
    df=df.drop_duplicates()
    # drop null values
    df.dropna(inplace=True)
    return df
    
def save_data(df, database_filename):
    from sqlalchemy import create_engine
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponce', engine, index=False) 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        messages,categories=load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(messages,categories)
        
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