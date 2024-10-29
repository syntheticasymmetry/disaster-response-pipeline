import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load messages and categories datasets, merge them on 'id'.

    Inputs:
    messages_filepath(str): Filepath for the messages dataset.
    categories_filepath (str): Filepath for the categories dataset.

    Returns:
    pd.DataFrame: Merged dataset containing both messages and categories.
    """
    # load messages dataset
    messages = pd.read_csv(messages_filepath)

    # load categories dataset
    categories = pd.read_csv(categories_filepath)

    # merge datasets
    df = pd.merge(messages, categories, on='id')

    return df


def clean_data(df):
    """
    Clead the merged dataset by splitting categories into separate columns, converting values to binary and removing duplicates.

    Inputs:
    df (pd.DataFrame): Merged dataset with the messages and categories.

    Returns:
    pd.DataFrame: Cleaned dataset with separate category columns and no duplicates.
    """
    # split categories into separate category columns
    categories = df['categories'].str.split(';', expand=True)

    # use the first row to extract new column names
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames

    # convert category values to just 0 or 1
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1].astype(int)
    
    # remove rows where 'related' is 2
    categories = categories[categories['related'] !=2]

    # filter 'df' based on the filtered categories indices to retain only rows with "related" equals to 0 or 1
    df = df.loc[categories.index]

    # drop the original categories column from df
    df = df.drop('categories', axis=1)

    # concatenate the original dataframe with the new 'categories' dataframe
    df = pd.concat([df, categories], axis=1)

    # remove duplucates
    df = df.drop_duplicates()

    return df


def save_data(df, database_filename):
    """
    Save the cleaned data to a SQLite database.

    Inputs:
    df (pd.DataFrame): Cleaned dataset.
    database_filename (str): Name of the AQLite database file to save the data.

    Returns:
    None
    """  
    # create SQLite engine
    engine = create_engine(f'sqlite:///{database_filename}')

    # save  df to SQLite database
    df.to_sql('DisasterMessages', engine, index=False, if_exists='replace')


def main():
    """
    1. Loads data from the messages and categories CSV files.
    2. Cleans the merged data by splitting categories into binary columns and removing duplicates.
    3. Saves the cleaned data to an SQLite database.

    Command-line inputs:
    sys.argv[1]: Filepath for the messages CSV file.
    sys.argv[2]: Filepath for the categories CSV file.
    sys.argv[3]: Filepath for the SQLite database to save the cleaned data.

    Example:
    python process_data.py messages.csv categories.csv DisasterResponse.db

    If the correct number if arguments is not provided, an error message is displayed with usage instructions.    
    """
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