def feature_engineer(df):
    """Perform the pre-established feature engineering on the
    provided dataframe

    Args:
        df (pandas.DataFrame): dataframe containing the dataset

    Returns:
        pandas.DataFrame: dataframe containing the dataset after performing the feature engineering
    """
    # Remove the 'Ticket' and 'Cabin' columns
    df = df.drop(["Ticket", "Cabin"], axis=1)
    df = df.dropna()
    return df
