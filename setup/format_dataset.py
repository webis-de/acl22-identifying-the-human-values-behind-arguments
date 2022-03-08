import pandas as pd

def combine_columns(df_arguments, df_labels):
    df = pd.merge(df_arguments, df_labels, on='Argument ID')
    
    return df

def split_arguments(df_arguments):
    train_arguments = df_arguments.loc[df_arguments['Usage'] == 'train'].drop(['Usage'], axis=1)
    train_arguments.reset_index(drop=True)
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1)
    valid_arguments.reset_index(drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1)
    test_arguments.reset_index(drop=True)
    
    return train_arguments, valid_arguments, test_arguments
