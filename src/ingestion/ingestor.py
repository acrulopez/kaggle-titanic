import pandas as pd


def ingest_data(dataset_path):
    """Loads the data from a csv file given its local path

    Args:
        dataset_path (str): path to the .csv file

    Returns:
        pandas.DataFrame: dataframe containing the dataset
    """
    return pd.read_csv(dataset_path)
