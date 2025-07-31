import pandas as pd
import os 
from sklearn.model_selection import train_test_split
import logging
import yaml

# ensure the "logs " directory exists
log_dir='logs'
os.makedirs(log_dir, exist_ok=True)

#logging configuration

# Create a logger instance
#  Creates or retrieves a logger with the name 'data_ingestion'
# Named loggers help in distinguishing logs from different modules
logger=logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')
# DEBUG < INFO < WARNING < ERROR < CRITICAL .... meaning the logger will capture all messages at DEBUG level and above


# Sends logs to the console/terminal (standard output).
console_hander=logging.StreamHandler()
console_hander.setLevel('DEBUG')


# Creates a file handler that writes logs to data_ingestion.log file
log_file_path=os.path.join(log_dir,'data_ingestion.log')
file_handler=logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter=logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_hander.setFormatter(formatter)
file_handler.setFormatter(formatter)
# Sets the previously defined formatting style to both handlers (console and file).

# You attach both handlers to the logger so it outputs logs to both file and console
logger.addHandler(console_hander)
logger.addHandler(file_handler)


# loading parameters from a YAML file
def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise


def load_data(data_url:str)->pd.DataFrame:
    """
    Load data from a CSV file located at the specified URL.
    
    Parameters:
    data_url (str): The URL or path to the CSV file.
    
    Returns:
    pd.DataFrame: DataFrame containing the loaded data.
    """
    try:
        logger.debug(f"Loading data from {data_url}")
        df = pd.read_csv(data_url)
        logger.debug("Data loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Basic data processing function."""
    try:
        df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'], inplace=True)
        df.rename(columns={'v1':'target', 'v2':'text'}, inplace=True)
        logger.debug("Data processed successfully")
        return df
    except KeyError as e:
        # A KeyError is raised when you try to access a dictionary key or a DataFrame column that does not exist
        logger.error('missing columns in the DataFrame: %s', e)
        # %s is a format specifier replaced by the actual missing key stored in e.
        raise
    except Exception as e:
        logger.error(f"Error processing data: {e}")
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str)-> None:
    """
    Save the train and test datasets to the specified directory.
    
    Parameters:
    train_data (pd.DataFrame): The training dataset.
    test_data (pd.DataFrame): The testing dataset.
    output_dir (str): Directory where the datasets will be saved.
    """
    try:
        raw_data_path=os.path.join(data_path,'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)
        
        logger.debug("train and test data saved successfully to %s", raw_data_path)
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        raise


def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']

    

        data_path="https://raw.githubusercontent.com/vikashishere/Datasets/main/spam.csv"
        df=load_data(data_url=data_path)
        final_df=preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data=train_data, test_data=test_data, data_path='data')
    except Exception as e:
        logger.error(f"An error occurred in the main function: {e}")
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
    logger.info("Data ingestion completed successfully.")