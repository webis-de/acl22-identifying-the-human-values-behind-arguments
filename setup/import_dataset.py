import os
import traceback
import pandas as pd


"""
Reads in a .tsv with pandas through a given path and returns it as a DataFrame object
:param dirname:     name of the directory which stores the .tsv
:param filename:    name of the .tsv
:return:            return of the .tsv as a DataFrame
"""
def load_tsv_dataframe(dirname, filename):
    CURR_DIR = os.getcwd()
    try:
        path = os.path.join(CURR_DIR, dirname, filename)
        dataframe = pd.read_csv(path, encoding='utf-8', sep='\t', header=0)
        return dataframe
    except IOError:
        traceback.print_exc()
