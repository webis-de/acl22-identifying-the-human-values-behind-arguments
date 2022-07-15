"""
    Collection of machine learning functions regarding the models:
    Bert,
    Support Vector Machine (SVM),
    1-Baseline

    Functions
    ---------
    train_bert_model(train_dataframe, model_dir, labels, test_dataframe=None, num_train_epochs=20):
        Train Bert model
    predict_bert_model(dataframe, model_dir, labels):
        Predict with Bert model
    train_svm(train_dataframe, labels, vectorizer_file, model_file, test_dataframe=None):
        Train Support Vector Machines (SVMs)
    predict_svm(dataframe, labels, vectorizer_file, model_file):
        Predict with Support Vector Machines (SVMs)
    predict_one_baseline(dataframe, labels):
        Predict with 1-Baseline model
    """
from .bert import (train_bert_model, predict_bert_model)
from .svm import (train_svm, predict_svm)
from .one_baseline import (predict_one_baseline)
