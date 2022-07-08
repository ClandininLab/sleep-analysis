""" Module for parsing data
"""

import pandas as pd


def read_data(data_path):
    """ Parses data into DataFrame

    Args:
      data_path: path to data file - str

    Returns:
      data_df - pd.DataFrame
    """
    df = pd.read_csv(
        data_path,
        sep='	',
        header=None,
        usecols=[1, 2] + list(range(9, 42)),
        parse_dates=[[1, 2]],
        index_col='1_2'
    )
    df.columns = ['light'] + [f'fly_{idx}' for idx in range(32)]
    df.index.name = 'date'
    df = df.astype('int32')

    return df


def write_data(data_path, data_df):
    """ Write DataFrame as CSV

    Args:
      data_path - str
      data_df: as returned by read_data, not zeitgeber_df - pd.DataFrame

    """
    pass
