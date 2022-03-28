import pandas as pd


def combine_columns(df_arguments, df_labels):
    """Combines the two `DataFrames` on column `Argument ID`"""
    return pd.merge(df_arguments, df_labels, on='Argument ID')


def split_arguments(df_arguments):
    """Splits `DataFrame` by column `Usage` into `train`-, `validation`-, and `test`-arguments"""
    train_arguments = df_arguments.loc[df_arguments['Usage'] == 'train'].drop(['Usage'], axis=1).reset_index(drop=True)
    valid_arguments = df_arguments.loc[df_arguments['Usage'] == 'validation'].drop(['Usage'], axis=1).reset_index(drop=True)
    test_arguments = df_arguments.loc[df_arguments['Usage'] == 'test'].drop(['Usage'], axis=1).reset_index(drop=True)
    
    return train_arguments, valid_arguments, test_arguments


def create_dataframe_head(argument_ids, model_name):
    """
        Creates `DataFrame` usable to append predictions to it

        Parameters
        ----------
        argument_ids : list[str]
            First column of the resulting DataFrame
        model_name : str
            Second column of DataFrame will contain the given model name

        Returns
        -------
        pd.DataFrame
            prepared DataFrame
    """
    df_model_head = pd.DataFrame(argument_ids, columns=['Argument ID'])
    df_model_head['Method'] = [model_name] * len(argument_ids)

    return df_model_head
