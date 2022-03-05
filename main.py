import transformers
import datasets

import torch
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          PreTrainedModel, BertModel, BertForSequenceClassification,
                          TrainingArguments, Trainer)
from transformers.modeling_outputs import SequenceClassifierOutput

from sklearn.metrics import f1_score
import numpy as np
import pandas as pd

print(f"Running on transformers v{transformers.__version__} and datasets v{datasets.__version__}")

f = open("output.txt", "w")
f.write(f"Running on transformers v{transformers.__version__} and datasets v{datasets.__version__}")
f.close()
