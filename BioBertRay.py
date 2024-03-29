#
# Finetune BERT for NER in medical data
# RayTune for hyperparameter finetuning
#
import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
import torch
from torch import nn
from ast import literal_eval
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Value, load_dataset, concatenate_datasets

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

bc5cdr = load_dataset("tner/bc5cdr")
ncbi = load_dataset("ncbi_disease")
mimic = load_dataset('csv', data_files="Resources/BioNLP_dataset.csv")
mimic = mimic['train'].train_test_split(test_size=0.2, seed=42)
print(mimic)
test_valid = mimic['test'].train_test_split(test_size=0.5, seed=42)
mimic = DatasetDict({
    'train': mimic['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})

# Adapt BC5CDR
#Rework chemical tags bc5cdr
def rework_tags(ex):
    for i, tags in enumerate(ex['tags']):
        for j, tag in enumerate(tags):
            if tag == 1:
                ex["tags"][i][j] = 0
            elif tag == 2:
                ex["tags"][i][j] = 1
            elif tag == 3:
                ex["tags"][i][j] = 2
            elif tag == 4:
                ex["tags"][i][j] = 0
    return ex

bc5cdr = bc5cdr.map(rework_tags, batched=True)
#bc5cdr = bc5cdr.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))
bc5cdr = bc5cdr.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Disease", "I-Disease"], id=None), length=-1, id=None))
bc5cdr = bc5cdr.filter(lambda example: len(example["tags"]) > 0)
bc5cdr = bc5cdr.filter(lambda ex: 1 in ex['tags'] or 2 in ex['tags'])
# Adapt NCBI
ncbi = ncbi.filter(lambda example: len(example["ner_tags"]) > 0)
ncbi = ncbi.remove_columns('id')
ncbi = ncbi.rename_column("ner_tags", "tags")
#ncbi = ncbi.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))
ncbi = ncbi.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Disease", "I-Disease"], id=None), length=-1, id=None))

#Adapt tags to bc5cdr (1 -> 2, 2 -> 3)
#def change_tags(ex):
#    for i, tags in enumerate(ex['tags']):
#        for j, tag in enumerate(tags):
#            if tag == 1:
#                ex["tags"][i][j] = 2
#            else:
#                if tag == 2:
#                    ex["tags"][i][j] = 3
#    return ex

#ncbi = ncbi.map(change_tags, batched=True)

# Adapt MIMIC don't needed here already been done in preprocessing
def int_tags(ex):
    for i, tags in enumerate(ex['tags']):
        ex['tags'][i] = literal_eval(ex['tags'][i])
        ex['tokens'][i] = literal_eval(ex['tokens'][i])
    return ex

mimic = mimic.map(int_tags,batched=True)
mimic = mimic.cast_column('tokens', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Disease", "I-Disease"], id=None), length=-1, id=None))
mimic = mimic.filter(lambda example: len(example["tags"]) > 0)


# Merge
datasets = DatasetDict()
datasets['train'] = concatenate_datasets([bc5cdr['train'],ncbi['train']])#,mimic['train']])
datasets['validation'] = concatenate_datasets([bc5cdr['validation'],ncbi['validation']])#,mimic['validation']])
datasets['test'] = concatenate_datasets([bc5cdr['test'],ncbi['test']])#,mimic['test']])
print(datasets)

from sklearn.utils import class_weight

class_weights=class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(np.concatenate(datasets['train']['tags'])),y=np.concatenate(datasets['train']['tags']))
class_weights=torch.tensor(class_weights,dtype=torch.float)

print('Class weights:')
print(class_weights)


labels_bio = ["O", "B-Disease", "I-Disease"]

# Tokenize and adapt datasets to tokenization
#tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def tokenize_and_realign(ex):
    tokenized_ex = tokenizer(ex["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(ex["tags"]):
        word_ids = tokenized_ex.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[previous_word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_ex["labels"] = labels
    return tokenized_ex

tokenized_dataset = datasets.map(tokenize_and_realign, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['tokens', 'tags'])
tokenized_mimic = mimic.map(tokenize_and_realign, batched=True)
tokenized_mimic = tokenized_mimic.remove_columns(['tokens', 'tags'])

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

example = datasets['train'][0]
labels = [labels_bio[i] for i in example["tags"]]

def compute_metrics(p):
    logits, labels = p
    predictions = np.argmax(logits, axis=-1)

    true_predictions = [
        [labels_bio[p] for (p, l) in zip(prediction, label) if l != -100]#(l != -100 and l != 0)]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_bio[l] for l in label if l != -100]#(l != -100 and l != 0)]
        for label in labels
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

id2label = {0:"O", 1:"B-Disease", 2:"I-Disease"}
label2id = {"O":0, "B-Disease":1, "I-Disease":2}

#model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", id2label=id2label, label2id=label2id)
#model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", id2label=id2label, label2id=label2id)
#model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner", num_labels=3, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
def model_init(trial):
    return AutoModelForTokenClassification.from_pretrained(
        "bert-base-uncased", 
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id, 
        ignore_mismatched_sizes=True
    )

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=1,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=50,
    eval_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune.search.hyperopt import HyperOptSearch

def ray_hp_space(trial):
    return {
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.2),
        #"num_train_epochs": [1, 2],
    }

scheduler = PopulationBasedTraining(time_attr='training_iteration', metric='eval_loss', mode='min', # metric='objective', mode='max',
    perturbation_interval=2,
    hyperparam_mutations={
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.2),
        #"num_train_epochs": [1, 2],
    })

#best_trial = trainer.hyperparameter_search(
#    direction="maximize",
#    backend="ray",
#    hp_space=ray_hp_space,
#    n_trials=4,
#    keep_checkpoints_num=1,
#    local_dir="~/BioNLP/ray_results/BioNER/",
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    #search_alg=HyperOptSearch(metric="objective", mode="max"),
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
#    scheduler=scheduler
    #compute_objective=compute_objective,
#)

#print(best_trial)


import os
path = os.path.dirname(os.path.abspath('model/ner/config.json'))

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
def model_init(trial):
    return AutoModelForTokenClassification.from_pretrained(
        path,
        local_files_only=True,
        num_labels=3, 
        id2label=id2label, 
        label2id=label2id, 
        ignore_mismatched_sizes=True
    )

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
    model=None,
    args=training_args,
    train_dataset=tokenized_mimic["train"],
    eval_dataset=tokenized_mimic["validation"],
    tokenizer=tokenizer,
    model_init=model_init,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

from ray import tune

def ray_hp_space(trial):
    return {
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.2),
        "per_device_train_batch_size": tune.choice([4, 8]),
        "num_train_epochs": tune.choice([4, 8]),
    }

scheduler = PopulationBasedTraining(time_attr='training_iteration', metric='objective', mode='max', #metric='eval_loss', mode='min',
    perturbation_interval=1,
    hyperparam_mutations={
        "learning_rate": tune.uniform(1e-5, 5e-5),
        "weight_decay": tune.uniform(0.0, 0.2),
        "per_device_train_batch_size": [4, 8],
        "num_train_epochs": [2, 3]
    })

best_trial = trainer.hyperparameter_search(
    direction="maximize",
    backend="ray",
    hp_space=ray_hp_space,
    n_trials=4,
    keep_checkpoints_num=1,
    local_dir="~/BioNLP/ray_results/MIMIC/",
    # https://docs.ray.io/en/latest/tune/api_docs/suggestion.html
    #search_alg=HyperOptSearch(metric="objective", mode="max"),
    # https://docs.ray.io/en/latest/tune/api_docs/schedulers.html
    scheduler=scheduler
    #compute_objective=compute_objective,
)

print(best_trial)
