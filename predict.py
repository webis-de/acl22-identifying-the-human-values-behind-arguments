import sys
import getopt
import os
import pandas as pd

from setup import (load_json_file, load_arguments_from_tsv, split_arguments,
                   write_tsv_dataframe, create_dataframe_head)
from models import (predict_bert_model, predict_one_baseline, predict_svm)

help_string = '\nUsage:  predict.py [OPTIONS]' \
              '\n' \
              '\nRequest prediction of the BERT model (and optional SVM / 1-Baseline) for all test arguments' \
              '\n' \
              '\nOptions:' \
              '\n  -a, --argument-dir string  Directory with the argument files (default' \
              '\n                             WORKING_DIR/data/)' \
              '\n  -h, --help                 Display help text' \
              '\n  -m, --model-dir string     Directory of the trained models (default' \
              '\n                             WORKING_DIR/data/models/)' \
              '\n  -o, --one-baseline         Request prediction of 1-Baseline model' \
              '\n  -s, --svm                  Request prediction of SVM'


def main(argv):
    # default values
    curr_dir = os.getcwd()
    model_dir = os.path.join(curr_dir, 'data/models/')
    run_svm = False
    run_one_baseline = False
    argument_dir = os.path.join(curr_dir, 'data/')

    try:
        opts, args = getopt.gnu_getopt(argv, "a:hm:os", ["argument-dir", "help", "model-dir=", "one-baseline", "svm"])
    except getopt.GetoptError:
        print(help_string)
        sys.exit(2)
    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print(help_string)
            sys.exit()
        elif opt in ('-a', '--argument-dir'):
            argument_dir = arg
        elif opt in ('-m', '--model-dir'):
            model_dir = arg
        elif opt in ('-o', '--one-baseline'):
            run_one_baseline = True
        elif opt in ('-s', '--svm'):
            run_svm = True

    # Check argument directory
    if not os.path.isdir(argument_dir):
        print('The specified argument directory "%s" does not exist' % argument_dir)
        sys.exit(2)
    # TODO: argument files are not present

    if not os.path.isdir(model_dir):
        print('The specified <model-dir> "%s" does not exist' % model_dir)
        sys.exit(2)
    # TODO: model save files not present

    # load arguments
    df_arguments = load_arguments_from_tsv(os.path.join(argument_dir, 'arguments.tsv'))

    value_json = load_json_file(os.path.join(argument_dir, 'values.json'))

    levels = value_json['level']
    num_levels = len(levels)

    # format dataset
    _, _, df_test = split_arguments(df_arguments)

    # predict with Bert model
    if run_svm or run_one_baseline:
        df_prediction = create_dataframe_head(df_test['Argument ID'], model_name='Bert')
    else:
        df_prediction = create_dataframe_head(df_test['Argument ID'])

    for i in range(num_levels):
        print("===> Bert: Predicting Level %s..." % levels[i])
        result = predict_bert_model(df_test,
                                    os.path.join(model_dir, 'bert_train_level' + levels[i]), len(value_json[levels[i]]))
        bert_prediction = 1 * (result.predictions > 0.5)
        df_prediction = pd.concat([df_prediction, pd.DataFrame(bert_prediction, columns=value_json[levels[i]])], axis=1)

    # predict with SVM
    if run_svm:
        df_svm = create_dataframe_head(df_test['Argument ID'], model_name='SVM')
        for i in range(num_levels):
            print("===> SVM: Predicting Level %s..." % levels[i])
            result = predict_svm(df_test, value_json[levels[i]],
                                 os.path.join(model_dir, 'svm/svm_train_level' + levels[i] + '.sav'))
            df_svm = pd.concat([df_svm, result], axis=1)

        df_prediction = pd.concat([df_prediction, df_svm])

    # predict with 1-Baseline
    if run_one_baseline:
        df_one_baseline = create_dataframe_head(df_test['Argument ID'], model_name='1-Baseline')
        for i in range(num_levels):
            print("===> 1-Baseline: Predicting Level %s..." % levels[i])
            result = predict_one_baseline(df_test, value_json[levels[i]])
            df_one_baseline = pd.concat([df_one_baseline, result], axis=1)

        df_prediction = pd.concat([df_prediction, df_one_baseline])

    # write predictions
    print("===> Writing predictions...")
    write_tsv_dataframe(os.path.join(argument_dir, 'predictions.tsv'), df_prediction)


if __name__ == '__main__':
    main(sys.argv[1:])
