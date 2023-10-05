import pandas as pd

def import_data():
    '''
    Function to import the dataset files for the Earthquake competition.
    Returns dataframes in following order:
        train_values
        train_labels
        test_values 
    '''
    train_values = pd.read_csv('../../data/raw/train_values.csv')
    train_labels = pd.read_csv('../../data/raw/train_labels.csv')
    test_values = pd.read_csv('../../data/raw/test_values.csv')
    return train_values, train_labels, test_values