import numpy as np
import pandas as pd


def predict_one_baseline(dataframe, labels):
    return pd.DataFrame(np.full((len(dataframe), len(labels)), 1, dtype=int), columns=labels)
