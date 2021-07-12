import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    Load messages and categories from csv
    '''
    messages = pd.read_csv(messages_filepath, index_col=0)
    categories = pd.read_csv(categories_filepath, index_col=0)
    df = messages.merge(categories, left_index=True, right_index=True)
    return df


def clean_data(df):
    '''
    Clean the categories column and drop duplicates
    '''
    categories = df['categories'].str.split(';', expand=True)  # Create DataFrame of 36 individual category columns
    category_colnames = categories.loc[2, :].str.split('-', expand=True)[0]
    categories.columns = category_colnames  # rename the columns of `categories`

    for column in categories:
        # set each value to be the last character of the string and convert to numeric
        categories[column] = categories[column].str.split('-', expand=True)[1].astype(int)

    categories.replace({2: 1}, inplace=True)

    df = df.drop(['categories', 'original'], axis=1)
    df = pd.concat([df, categories], axis=1)

    duplicates = df['message'].duplicated()
    # drop duplicates
    df = df[~duplicates]
    return df


def save_data(df, database_filename):
    '''
    Export the cleaned data to sqlite database
    '''
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response', engine, index=False, if_exists='replace')


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