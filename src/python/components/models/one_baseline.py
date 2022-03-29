import numpy as np
import pandas as pd


def predict_one_baseline(dataframe, labels):
    """
        Classifies each argument in the dataframe as corresponding to all labels.

        Parameters
        ----------
        dataframe : pd.DataFrame
            The arguments to be classified
        labels : list[str]
            The listing of all labels

        Returns
        -------
        DataFrame
            the predictions given by the model
        """
    return pd.DataFrame(np.full((len(dataframe), len(labels)), 1, dtype=int), columns=labels)
