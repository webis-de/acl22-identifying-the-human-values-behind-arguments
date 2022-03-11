import os
import traceback
import csv


def write_tsv_dataframe(filepath, dataframe):
    try:
        dataframe.to_csv(filepath, encoding='utf-8', sep='\t', index=False, header=True, quoting=csv.QUOTE_NONE)
    except IOError:
        traceback.print_exc()
