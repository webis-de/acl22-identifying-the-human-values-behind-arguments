import traceback
import pandas as pd
import json


class MissingColumnError(AttributeError):
    """Error indicating that an imported dataframe lacks necessary columns"""
    pass


def load_json_file(filepath):
    return json.load(open(filepath))


def load_arguments_from_tsv(filepath):
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        if not {'Argument ID', 'Premise'}.issubset(set(dataframe.columns.values)):
            raise MissingColumnError('The argument "%s" file does not contain the minimum required columns [Argument ID, Premise].' % filepath)
        if 'Usage' not in dataframe.columns.values:
            dataframe['Usage'] = ['test'] * len(dataframe)
        return dataframe
    except IOError:
        traceback.print_exc()


def load_labels_from_tsv(filepath, label_order):
    try:
        dataframe = pd.read_csv(filepath, encoding='utf-8', sep='\t', header=0)
        dataframe = dataframe[['Argument ID'] + label_order]
        return dataframe
    except IOError:
        traceback.print_exc()
    except KeyError:
        raise MissingColumnError('The file "%s" does not contain the required columns for its level.' % filepath)
