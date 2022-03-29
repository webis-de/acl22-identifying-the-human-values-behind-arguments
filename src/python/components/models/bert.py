import torch

from datasets import (Dataset, DatasetDict, load_dataset)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from sklearn.metrics import f1_score

import numpy as np


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    """Compute accuracy of predictions"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=True):
    """Compute label-wise and averaged F1-scores"""
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i], zero_division=0), 2)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    """Custom metric calculation function for MultiLabelTrainer"""
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'marco-avg-f1score': f1scores['avg-f1-score']}


class MultiLabelTrainer(Trainer):
    """
        A transformers `Trainer` with custom loss computation

        Methods
        -------
        compute_loss(model, inputs, return_outputs=False):
            Overrides loss computation from Trainer class
        """
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss computation"""
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def tokenize_and_encode(examples):
    """Tokenizes each arguments "Premise" """
    return tokenizer(examples['Premise'], truncation=True)


def convert_to_dataset(train_dataframe, test_dataframe, labels):
    """
        Converts pandas DataFrames into a DatasetDict

        Parameters
        ----------
        train_dataframe : pd.DataFrame
            Arguments to be listed as "train"
        test_dataframe : pd.DataFrame
            Arguments to be listed as "test"
        labels : list[str]
            The labels in both DataFrames

        Returns
        -------
        tuple(DatasetDict, list[str])
            a `DatasetDict` with attributes "train" and "test" for the listed arguments,
            a `list` with the contained labels
        """
    column_intersect = [x for x in (['Premise'] + labels) if x in train_dataframe.columns.values]

    train_dataset = Dataset.from_dict((train_dataframe[column_intersect]).to_dict('list'))
    test_dataset = Dataset.from_dict((test_dataframe[column_intersect]).to_dict('list'))

    ds = DatasetDict()
    ds['train'] = train_dataset
    ds['test'] = test_dataset

    ds = ds.map(lambda x: {"labels": [int(x[c]) for c in ds['train'].column_names if
                                      c not in ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Part']]})

    cols = ds['train'].column_names
    cols.remove('labels')

    ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols)

    cols.remove('Premise')

    return ds_enc, cols


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def load_model_from_data_dir(model_dir, num_labels):
    """Loads Bert model from specified directory and converts to CUDA model if available"""
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=num_labels)
    if torch.cuda.is_available():
        return model.to('cuda')
    return model


def predict_bert_model(dataframe, model_dir, labels):
    """
        Classifies each argument using the Bert model stored in `model_dir`

        Parameters
        ----------
        dataframe: pd.Dataframe
            The arguments to be classified
        model_dir: str
            The directory of the pre-trained Bert model to use
        labels: list[str]
            The labels to predict

        Returns
        -------
        np.ndarray
            numpy nd-array with the predictions given by the model
        """
    ds, no_labels = convert_to_dataset(dataframe, dataframe, labels)
    num_labels = len(labels)
    ds = ds.remove_columns(['labels'])

    batch_size = 8
    args = TrainingArguments(
        output_dir=model_dir,
        do_train=False,
        do_eval=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size
    )

    model = load_model_from_data_dir(model_dir, num_labels=num_labels)

    multi_trainer = MultiLabelTrainer(
        model,
        args,
        tokenizer=tokenizer
    )

    prediction = 1 * (multi_trainer.predict(ds['train']).predictions > 0.5)

    return prediction


def train_bert_model(train_dataframe, model_dir, labels, test_dataframe=None, num_train_epochs=20):
    """
        Trains Bert model with the arguments in `train_dataframe`

        Parameters
        ----------
        train_dataframe: pd.DataFrame
            The arguments to be trained on
        model_dir: str
            The directory for storing the trained model
        labels : list[str]
            The labels in the training data
        test_dataframe: pd.DataFrame, optional
            The validation arguments (default is None)
        num_train_epochs: int, optional
            The number of training epochs (default is 20)

        Returns
        -------
        Metrics
            result of validation if `test_dataframe` is not None
        NoneType
            otherwise
        """
    if test_dataframe is None:
        test_dataframe = train_dataframe
    ds, labels = convert_to_dataset(train_dataframe, test_dataframe, labels)

    batch_size = 8

    args = TrainingArguments(
        output_dir=model_dir,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='marco-avg-f1score'
    )

    model = load_model_from_data_dir("bert-base-uncased", num_labels=len(labels))

    multi_trainer = MultiLabelTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=lambda x: compute_metrics(x, labels),
        tokenizer=tokenizer
    )

    multi_trainer.train()

    model.save_pretrained(model_dir)

    if test_dataframe is not None:
        return multi_trainer.evaluate()
