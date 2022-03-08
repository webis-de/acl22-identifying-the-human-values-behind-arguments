import numpy as np
import os
import traceback
import pandas as pd

from transformers import (AutoModelForSequenceClassification,
                          BertForSequenceClassification)
from transformers.modeling_outputs import SequenceClassifierOutput

from setup import (load_tsv_dataframe, combine_columns, split_arguments, write_tsv_dataframe)
from models import (train_bert_model, predict_bert_model)


def main():
    # get arguments
    df_arguments = load_tsv_dataframe('data', 'arguments.tsv')

    levels = ['1', '2', '3', '4a', '4b']
    num_levels = len(levels)

    # format dataset
    df_train_all = []
    df_valid_all = []
    df_test_all = []
    level_labels = []
    for i in range(num_levels):
        # read labels from .tsv file
        df_labels = load_tsv_dataframe('data', 'labels-level' + levels[i] + '.tsv')
        # save list of labels for each level
        level_labels.append(list((df_labels.drop(['Argument ID'], axis=1)).columns.values))
        # join arguments and labels
        df_full_level = combine_columns(df_arguments, df_labels)
        # split dataframe by usage
        train_arguments, valid_arguments, test_arguments = split_arguments(df_full_level)
        df_train_all.append(train_arguments)
        df_valid_all.append(valid_arguments)
        df_test_all.append(test_arguments)

    # Models

    # 1-Baseline

    # SVM

    # Bert
    # for i in [4]:
    for i in range(num_levels):
        bert_model = train_bert_model(df_train_all[i], df_valid_all[i], './data/bert_models/train_level' + levels[i])
        # bert_model = AutoModelForSequenceClassification.from_pretrained('./data/bert_models/train_level' + levels[i], num_labels=len(level_labels[i]))
        result = predict_bert_model(df_test_all[i], './data/bert_models/train_level' + levels[i], bert_model)
        print(result)
        for pred in result.label_ids:
            print(pred)

    # Combine predictions

    # Write results to predictions.tsv
    # write_tsv_dataframe('data', 'predictions.tsv', result)


if __name__ == '__main__':
    main()
