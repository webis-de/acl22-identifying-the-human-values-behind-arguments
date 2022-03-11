from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

import pickle
import pandas as pd
import numpy as np


class PickleError(AssertionError):
    """Error indicating that pickle retrieved unexpected code."""
    pass


def load_all_svms(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


def predict_svm(dataframe, labels, model_file):
    """

    :param dataframe:
    :param labels:
    :param model_file:
    :raises PickleError: if the specified model_file does not contain sklearn Pipelines
    :return:
    """
    input_vector = dataframe['Premise']
    df_model_predictions = {}

    for i, svm in enumerate(load_all_svms(model_file)):
        if not isinstance(svm, Pipeline):
            raise PickleError(
                'The data read from file "{}" via pickle does not have the expected type. To prevent the execution of potentially malicious code this portion of the execution is skipped.'.format(
                    model_file))
        df_model_predictions[labels[i]] = svm.predict(input_vector)

    return pd.DataFrame(df_model_predictions, columns=labels)


def train_svm(train_dataframe, labels, model_file, test_dataframe=None):
    train_input_vector = train_dataframe['Premise']
    if test_dataframe is not None:
        valid_input_vector = test_dataframe['Premise']
        f1_scores = {}

    with open(model_file, "wb") as f:
        for label_name in labels:
            svm = Pipeline([
                ('tfidf', TfidfVectorizer(stop_words='english')),
                ('clf', OneVsRestClassifier(LinearSVC(C=18, class_weight='balanced', max_iter=10000), n_jobs=1)),
            ])
            svm.fit(train_input_vector, train_dataframe[label_name])
            if test_dataframe is not None:
                valid_pred = svm.predict(valid_input_vector)
                f1_scores[label_name] = round(f1_score(test_dataframe[label_name], valid_pred, zero_division=0), 2)
            pickle.dump(svm, f)

    if test_dataframe is not None:
        f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)
        return f1_scores
    return None
