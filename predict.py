import sys
import getopt
import os
import pandas as pd

from components.python_components.setup import (load_json_file, load_arguments_from_tsv, split_arguments,
                                                write_tsv_dataframe, create_dataframe_head)
from components.python_components.models import (predict_bert_model, predict_one_baseline, predict_svm, PickleError)

help_string = '\nUsage:  predict.py [OPTIONS]' \
              '\n' \
              '\nRequest prediction of the BERT model (and optional SVM / 1-Baseline) for all test arguments' \
              '\n' \
              '\nOptions:' \
              '\n  -c, --classifier string  Select classifier: "b" for Bert, "s" for SVM, "o" for 1-Baseline,' \
              '\n                           or combination like "so" (default "b")' \
              '\n  -d, --data-dir string    Directory with the argument files (default' \
              '\n                           WORKING_DIR/data/)' \
              '\n  -h, --help               Display help text' \
              '\n  -m, --model-dir string   Directory of the trained models (default' \
              '\n                           WORKING_DIR/data/models/)'


def main(argv):
    # default values
    curr_dir = os.getcwd()
    model_dir = os.path.join(curr_dir, 'data/models/')
    run_bert = True
    run_svm = False
    run_one_baseline = False
    argument_dir = os.path.join(curr_dir, 'data/')

    try:
        opts, args = getopt.gnu_getopt(argv, "c:d:hm:", ["classifier=", "data-dir=", "help", "model-dir="])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help_string)
            sys.exit()
        elif opt in ('-c', '--classifier'):
            run_bert = 'b' in arg.lower()
            run_svm = 's' in arg.lower()
            run_one_baseline = 'o' in arg.lower()
            if not run_bert and not run_svm and not run_one_baseline:
                print('No classifiers selected')
                sys.exit(2)
        elif opt in ('-d', '--data-dir'):
            argument_dir = arg
        elif opt in ('-m', '--model-dir'):
            model_dir = arg
        elif opt in ('-o', '--one-baseline'):
            run_one_baseline = True
        elif opt in ('-s', '--svm'):
            run_svm = True

    # Check argument directory
    if not os.path.isdir(argument_dir):
        print('The specified data directory "%s" does not exist' % argument_dir)
        sys.exit(2)

    argument_filepath = os.path.join(argument_dir, 'arguments.tsv')
    value_json_filepath = os.path.join(argument_dir, 'values.json')

    if not os.path.isfile(argument_filepath):
        print('The required file "arguments.tsv" is not present in the argument directory')
        sys.exit(2)
    if not os.path.isfile(value_json_filepath):
        print('The required file "values.json" is not present in the argument directory')
        sys.exit(2)

    # load arguments
    df_arguments = load_arguments_from_tsv(argument_filepath)
    if len(df_arguments) < 1:
        print('There are no arguments in file "%s"' % argument_filepath)
        sys.exit(2)

    value_json = load_json_file(value_json_filepath)

    try:
        levels = value_json['level']
    except KeyError:
        print('Missing attribute "level" in value.json')
        sys.exit(2)
    num_levels = len(levels)

    # check levels
    for i in range(num_levels):
        if levels[i] not in value_json:
            print('Missing attribute "{}" in value.json'.format(levels[i]))
            sys.exit(2)

    # check model directory
    if not os.path.isdir(model_dir):
        print('The specified <model-dir> "%s" does not exist' % model_dir)
        sys.exit(2)

    for i in range(num_levels):
        if not os.path.exists(os.path.join(model_dir, 'bert_train_level{}'.format(levels[i]))):
            print('Missing saved Bert model for level "{}"'.format(levels[i]))
            sys.exit(2)
        if run_svm and not os.path.exists(os.path.join(model_dir, 'svm/svm_train_level{}.sav'.format(levels[i]))):
            print('Missing saved SVM models for level "{}"'.format(levels[i]))
            sys.exit(2)

    # format dataset
    _, _, df_test = split_arguments(df_arguments)

    if len(df_test) < 1:
        print('There are no arguments listed for prediction.')
        sys.exit()

    # predict with Bert model
    if run_bert:
        df_bert = create_dataframe_head(df_test['Argument ID'], model_name='Bert')
        for i in range(num_levels):
            print("===> Bert: Predicting Level %s..." % levels[i])
            result = predict_bert_model(df_test, os.path.join(model_dir, 'bert_train_level{}'.format(levels[i])),
                                        value_json[levels[i]])
            df_bert = pd.concat([df_bert, pd.DataFrame(result, columns=value_json[levels[i]])], axis=1)
        df_prediction = df_bert

    # predict with SVM
    if run_svm:
        try:
            df_svm = create_dataframe_head(df_test['Argument ID'], model_name='SVM')
            for i in range(num_levels):
                print("===> SVM: Predicting Level %s..." % levels[i])
                result = predict_svm(df_test, value_json[levels[i]],
                                     os.path.join(model_dir, 'svm/svm_train_level{}.sav'.format(levels[i])))
                df_svm = pd.concat([df_svm, result], axis=1)

            if not run_bert:
                df_prediction = df_svm
            else:
                df_prediction = pd.concat([df_prediction, df_svm])
        except PickleError as error:
            print(repr(error))

    # predict with 1-Baseline
    if run_one_baseline:
        df_one_baseline = create_dataframe_head(df_test['Argument ID'], model_name='1-Baseline')
        for i in range(num_levels):
            print("===> 1-Baseline: Predicting Level %s..." % levels[i])
            result = predict_one_baseline(df_test, value_json[levels[i]])
            df_one_baseline = pd.concat([df_one_baseline, result], axis=1)

        if not run_bert and not run_svm:
            df_prediction = df_one_baseline
        else:
            df_prediction = pd.concat([df_prediction, df_one_baseline])

    # write predictions
    print("===> Writing predictions...")
    write_tsv_dataframe(os.path.join(argument_dir, 'predictions.tsv'), df_prediction)


if __name__ == '__main__':
    main(sys.argv[1:])
