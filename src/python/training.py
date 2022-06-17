import sys
import getopt
import os

from components.setup import (load_values_from_json, load_arguments_from_tsv, load_labels_from_tsv,
                                                combine_columns, split_arguments)
from components.models import (train_bert_model, train_svm)

help_string = '\nUsage:  training.py [OPTIONS]' \
              '\n' \
              '\nTrain the BERT model (and optional SVM) on the arguments' \
              '\n' \
              '\nOptions:' \
              '\n  -c, --classifier string  Select classifier: "b" for Bert, "s" for SVM, "bs" for both (default' \
              '\n                           "b")' \
              '\n  -d, --data-dir string    Directory with the argument files (default "/data/")' \
              '\n  -h, --help               Display help text' \
              '\n  -l, --levels string      Comma-separated list of taxonomy levels to train models for (default' \
              '\n                           "1,2,3,4a,4b")' \
              '\n  -m, --model-dir string   Directory for saving the trained models (default "/models/")' \
              '\n  -v, --validate           Request evaluation after training'


def main(argv):
    # default values
    curr_dir = os.getcwd()
    run_bert = True
    run_svm = False
    data_dir = '/data/'
    levels = ["1", "2", "3", "4a", "4b"]
    model_dir = '/models/'
    validate = False

    try:
        opts, args = getopt.gnu_getopt(argv, "c:d:hl:m:v", ["classifier=", "data-dir=", "help", "levels=", "model-dir=", "validate"])
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
            if not run_bert and not run_svm:
                print('No classifiers selected')
                sys.exit(2)
        elif opt in ('-d', '--data-dir'):
            data_dir = arg
        elif opt in ('-l', '--levels'):
            levels = arg.split(",")
        elif opt in ('-m', '--model-dir'):
            model_dir = arg
        elif opt in ('-v', '--validate'):
            validate = True

    svm_dir = os.path.join(model_dir, 'svm')

    # Check data directory
    if not os.path.isdir(data_dir):
        print('The specified data directory "%s" does not exist' % data_dir)
        sys.exit(2)

    # Check model directory
    if os.path.isfile(model_dir):
        print('The specified <model-dir> "%s" points to an existing file' % model_dir)
        sys.exit(2)
    if os.path.isdir(model_dir) and len(os.listdir(model_dir)) > 0:
        print('The specified <model-dir> "%s" already exists and contains files' % model_dir)
        decision = input('Do You still want to proceed? [y/n]\n').lower()
        if decision != 'y':
            sys.exit(-1)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    if run_svm and not os.path.isdir(svm_dir):
        if os.path.exists(svm_dir):
            print('Unable to create svm directory at "%s"' % svm_dir)
        else:
            os.mkdir(svm_dir)

    argument_filepath = os.path.join(data_dir, 'arguments.tsv')
    value_json_filepath = os.path.join(data_dir, 'values.json')

    if not os.path.isfile(argument_filepath):
        print('The required file "arguments.tsv" is not present in the data directory')
        sys.exit(2)
    if not os.path.isfile(value_json_filepath):
        print('The required file "values.json" is not present in the data directory')
        sys.exit(2)

    # load arguments
    df_arguments = load_arguments_from_tsv(argument_filepath, default_usage='train')
    if len(df_arguments) < 1:
        print('There are no arguments in file "%s"' % argument_filepath)
        sys.exit(2)

    values = load_values_from_json(value_json_filepath)
    num_levels = len(levels)

    # check levels
    for i in range(num_levels):
        if levels[i] not in values:
            print('Missing attribute "{}" in value.json'.format(levels[i]))
            sys.exit(2)

    # format dataset
    df_train_all = []
    df_valid_all = []
    for i in range(num_levels):
        label_filepath = os.path.join(data_dir, 'labels-level{}.tsv'.format(levels[i]))
        if not os.path.isfile(label_filepath):
            print('The required file "labels-level{}.tsv" is not present in the data directory'.format(levels[i]))
            sys.exit(2)
        # read labels from .tsv file
        df_labels = load_labels_from_tsv(label_filepath, values[levels[i]])
        # join arguments and labels
        df_full_level = combine_columns(df_arguments, df_labels)
        # split dataframe by usage
        train_arguments, valid_arguments, _ = split_arguments(df_full_level)
        df_train_all.append(train_arguments)
        df_valid_all.append(valid_arguments)

    if len(df_train_all[0]) < 1:
        print('There are no arguments listed for training.')
        sys.exit()

    if validate and len(df_valid_all[0]) < 1:
        print('There are no arguments listed for validation. Proceeding without validation.')
        validate = False

    # train bert model
    if run_bert:
        for i in range(num_levels):
            print("===> Bert: Training Level %s..." % levels[i])
            if validate:
                bert_model_evaluation = train_bert_model(df_train_all[i],
                                                         os.path.join(model_dir, 'bert_train_level{}'.format(levels[i])),
                                                         values[levels[i]], test_dataframe=df_valid_all[i])
                print("F1-Scores for Level %s:" % levels[i])
                print(bert_model_evaluation['eval_f1-score'])
            else:
                train_bert_model(df_train_all[i], os.path.join(model_dir, 'bert_train_level{}'.format(levels[i])),
                                 values[levels[i]])

    if run_svm:
        for i in range(num_levels):
            print("===> SVM: Training Level %s..." % levels[i])
            if validate:
                svm_f1_scores = train_svm(df_train_all[i], values[levels[i]],
                                          os.path.join(model_dir, 'svm/svm_train_level{}_vectorizer.json'.format(levels[i])),
                                          os.path.join(model_dir, 'svm/svm_train_level{}_models.json'.format(levels[i])),
                                          test_dataframe=df_valid_all[i])
                print("F1-Scores for Level %s:" % levels[i])
                print(svm_f1_scores)
            else:
                train_svm(df_train_all[i], values[levels[i]],
                          os.path.join(model_dir, 'svm/svm_train_level{}_vectorizer.json'.format(levels[i])),
                          os.path.join(model_dir, 'svm/svm_train_level{}_models.json'.format(levels[i])))


if __name__ == '__main__':
    main(sys.argv[1:])
