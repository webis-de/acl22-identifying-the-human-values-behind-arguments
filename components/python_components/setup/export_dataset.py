import traceback
import csv

import pandas as pd


def write_tsv_dataframe(filepath, dataframe):
    """
        Stores `DataFrame` as tsv file

        Parameters
        ----------
        filepath : str
            Path to tsv file
        dataframe : pd.DataFrame
            DataFrame to store

        Raises
        ------
        IOError
            if the file can't be opened
    """
    try:
        dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()
