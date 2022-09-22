from patsy import dmatrices
import pandas as pd
import numpy as np
from constants import SEED


def set_seed(func):
    """Decorator function to set the seed before any algorithm
    in order to ensure reproducibility

    Args:
        seed (int, optional): seed to be set. Defaults to 0.
    """
    np.random.seed(SEED)
    return func


def split_x_y(df, formula):
    """Split the dataframe on y and x

    Args:
        df (pandas.DataFrame): dataframe containing the dataset
        formula (str): formula to apply to dmatrices

    Returns:
        pandas.DataFrame, pandas.DataFrame: dataframes of y and x
    """
    return dmatrices(formula, data=df, return_type="dataframe")
