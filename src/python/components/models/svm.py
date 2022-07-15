from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score

import pandas as pd
import numpy as np

import json

# constant label values
vocab_label = 'vocabulary'
idf_label = 'idf'
intercept_label = 'intercept'
coef_label = 'coef'


class MyLinearSVC(LinearSVC):
    """
        A class to load and store a pretrained linear svm for predictions

        ...
        Attributes
        ----------
        intercept : int
            The intercept constant
        coef : ndarray
            The coefficients for all observed features

        Methods
        -------
        predict(X):
            Overrides `predict(X)` from LinearSVC
    """

    def __init__(self, intercept, coef):
        """
            Constructs all necessary attributes for the MyLinearSVC object

            Parameters
            ----------
            intercept : int
                The intercept constant
            coef : list[int]
                The coefficients for all observed features
        """
        LinearSVC.__init__(self, C=18, class_weight='balanced', max_iter=10000)
        self.intercept = intercept
        self.coef = np.asarray(coef)
        self.size = len(coef)

    def predict(self, X):
        """
            Predict class labels for samples in X

            Parameters
            ----------
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                The data matrix for which we want to get the predictions

            Returns
            -------
            ndarray of shape (n_samples,)
                Vector containing the class labels for each sample
        """
        input = np.transpose(X.todense())
        matrix = np.squeeze(np.asarray(self.__my_predict(input)))
        return np.vectorize(lambda x: 1 if x >= 0.5 else 0)(matrix)

    def __my_predict(self, input):
        """
            Predict class label probability for samples in input

            Parameters
            ----------
            input : ndarray of shape (n_features, n_samples)
                The data matrix for which we want to get the predictions

            Returns
            -------
            ndarray of shape (n_samples,)
                Vector containing the class labels for each sample
        """
        result = input[0] * self.coef[0]
        for i in range(1, self.size):
            result += input[i] * self.coef[i]
        result += self.intercept
        return result


def predict_svm(dataframe, labels, vectorizer_file, model_file):
    """
        Classifies each argument in the dataframe using the trained Support Vector Machines (SVMs) in the `model_file`

        Parameters
        ----------
        dataframe : pd.DataFrame
            The arguments to be classified
        labels : list[str]
            The listing of all labels
        vectorizer_file : str
            The file containing the fitted data from the TfidfVectorizer
        model_file : str
            The file containing the serialized SVM models

        Returns
        -------
        DataFrame
            the predictions given by the model
        """
    input_vector = dataframe['Premise']
    df_model_predictions = {}

    # load vectorizer
    with open(vectorizer_file, "r") as f:
        vectorizer_json = json.load(f)

    vocabulary = vectorizer_json[vocab_label]
    idf = np.asarray(vectorizer_json[idf_label])

    vectorizer = TfidfVectorizer(vocabulary=vocabulary)
    vectorizer.idf_ = idf

    with open(model_file, "r") as f:
        model_json = json.load(f)

    for label_name in labels:
        model_dict = model_json[label_name]
        svm = Pipeline([
            ('tfidf', vectorizer),
            ('clf', MyLinearSVC(intercept=model_dict[intercept_label], coef=model_dict[coef_label])),
        ])
        df_model_predictions[label_name] = svm.predict(input_vector)

    return pd.DataFrame(df_model_predictions, columns=labels)


def train_svm(train_dataframe, labels, vectorizer_file, model_file, test_dataframe=None):
    """
        Trains Support Vector Machines (SVMs) on the arguments in the train_dataframe and saves them in `model_file`

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            The arguments to be trained on
        labels : list[str]
            The listing of all labels
        vectorizer_file : str
            The file for storing the fitted data from the TfidfVectorizer
        model_file : str
            The file for storing the serialized SVM models
        test_dataframe : pd.DataFrame, optional
            The validation arguments (default is None)

        Returns
        -------
        dict
            f1-scores of validation if `test_dataframe` is not None
        NoneType
            otherwise
        """
    train_input_vector = train_dataframe['Premise']
    if test_dataframe is not None:
        valid_input_vector = test_dataframe['Premise']
        f1_scores = {}

    vectorizer = TfidfVectorizer(stop_words='english')
    vectorizer.fit(train_input_vector)

    vectorizer_json = {vocab_label: vectorizer.vocabulary_, idf_label: vectorizer.idf_.tolist()}

    with open(vectorizer_file, "w") as f:
        json.dump(vectorizer_json, f)

    # dictionary for storing model data
    model_json = {}

    for label_name in labels:
        svm = Pipeline([
            ('tfidf', vectorizer),
            ('clf', OneVsRestClassifier(LinearSVC(C=18, class_weight='balanced', max_iter=10000), n_jobs=1)),
        ])
        svm.fit(train_input_vector, train_dataframe[label_name])

        coef = np.squeeze(np.asarray(svm.steps[1][1].estimators_[0].coef_)).tolist()
        intercept = svm.steps[1][1].estimators_[0].intercept_[0]
        model_json[label_name] = {intercept_label: intercept, coef_label: coef}

        if test_dataframe is not None:
            valid_pred = svm.predict(valid_input_vector)
            f1_scores[label_name] = round(f1_score(test_dataframe[label_name], valid_pred, zero_division=0), 2)

    with open(model_file, "w") as f:
        json.dump(model_json, f)

    if test_dataframe is not None:
        f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)
        return f1_scores
