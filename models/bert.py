import torch

from datasets import (Dataset, DatasetDict, load_dataset)
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import f1_score

import numpy as np
import pandas as pd


def accuracy_thresh(y_pred, y_true, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    # print(y_true.bool())
    # print((y_pred>thresh))
    return ((y_pred > thresh) == y_true.bool()).float().mean().item()


def f1_score_per_label(y_pred, y_true, value_classes, thresh=0.5, sigmoid=True):
    y_pred = torch.from_numpy(y_pred)
    y_true = torch.from_numpy(y_true)
    if sigmoid:
        y_pred = y_pred.sigmoid()

    y_true = y_true.bool().numpy()
    y_pred = (y_pred > thresh).numpy()

    f1_scores = {}
    for i, v in enumerate(value_classes):
        f1_scores[v] = round(f1_score(y_true[:, i], y_pred[:, i]), 2)

    f1_scores['avg-f1-score'] = round(np.mean(list(f1_scores.values())), 2)

    return f1_scores


def compute_metrics(eval_pred, value_classes):
    predictions, labels = eval_pred
    f1scores = f1_score_per_label(predictions, labels, value_classes)
    return {'accuracy_thresh': accuracy_thresh(predictions, labels), 'f1-score': f1scores,
            'marco-avg-f1score': f1scores['avg-f1-score']}


class MultilabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.BCEWithLogitsLoss()
        loss = loss_fct(logits.view(-1, self.model.config.num_labels),
                        labels.float().view(-1, self.model.config.num_labels))
        return (loss, outputs) if return_outputs else loss


def tokenize_and_encode(examples):
    return tokenizer(examples['Premise'], truncation=True)


def convert_to_dataset(train_dataframe, test_dataframe):
    train_dataset = Dataset.from_dict(train_dataframe.to_dict('list'))
    test_dataset = Dataset.from_dict(test_dataframe.to_dict('list'))

    ds = DatasetDict()
    ds['train'] = train_dataset
    ds['test'] = test_dataset

    ds = ds.remove_columns(['Conclusion', 'Stance', 'Part'])

    ds = ds.map(lambda x: {"labels": [int(x[c]) for c in ds['train'].column_names if
                                      c not in ['Argument ID', 'Conclusion', 'Stance', 'Premise', 'Part']]})

    cols = ds['train'].column_names
    cols.remove('labels')
    cols.remove('Argument ID')

    ds_enc = ds.map(tokenize_and_encode, batched=True, remove_columns=cols)

    cols.remove('Premise')

    return ds_enc, cols


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


def predict_bert_model(dataframe, output_dir, model):
    ds, labels = convert_to_dataset(dataframe, dataframe)
    batch_size = 8
    args = TrainingArguments(
        output_dir=output_dir,
        do_train=False,
        do_eval=False,
        do_predict=True,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=20,
        weight_decay=0.01
    )

    print("===> Predicting...")
    multi_trainer = MultilabelTrainer(
        model,
        args,
        train_dataset=ds['train'],
        eval_dataset=ds['train'],
        compute_metrics=lambda x: compute_metrics(x, labels),
        tokenizer=tokenizer
    )

    return multi_trainer.predict(ds['train'])


def train_bert_model(train_dataframe, test_dataframe, output_dir, num_train_epochs=1):
    ds, labels = convert_to_dataset(train_dataframe, test_dataframe)

    batch_size = 8

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="steps",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_train_epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model='marco-avg-f1score'
    )

    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=len(labels))

    print("===> Training...")
    multi_trainer = MultilabelTrainer(
        model,
        args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        compute_metrics=lambda x: compute_metrics(x, labels),
        tokenizer=tokenizer
    )

    multi_trainer.train()

    model.save_pretrained(output_dir)

    return model