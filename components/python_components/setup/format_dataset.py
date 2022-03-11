import pandas as pd


def combine_columns(df_arguments, df_labels):
    return pd.merge(df_arguments, df_labels, on='Argument ID')


def split_arguments(df_arguments):
    train_arguments = df_arguments.loc[df_arguments['Usage'] == 'train'].drop(['Usage'], axis=1).reset_index(drop=True)
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1).reset_index(drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1).reset_index(drop=True)
    
    return train_arguments, valid_arguments, test_arguments


def create_dataframe_head(argument_ids, model_name=''):
    df_model_head = pd.DataFrame(argument_ids, columns=['Argument ID'])

    if model_name != '':
        df_model_head['Method'] = [model_name] * len(argument_ids)

    return df_model_head
