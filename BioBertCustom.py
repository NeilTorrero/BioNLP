#
# Finetune BERT for NER in medical data
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

mimic = load_dataset('csv', data_files="Resources/BioNLP2_dataset1.csv")
mimic = mimic['train'].train_test_split(test_size=0.2)
test_valid = mimic['test'].train_test_split(test_size=0.5)
mimic = DatasetDict({
    'train': mimic['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})
print(mimic)

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

labels_bio = ["O", "B-Disease", "I-Disease"]

# Tokenize and adapt datasets to tokenization
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
#tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
#tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
example = mimic['train'][0]
tokenized = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])

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

tokenized_mimic = mimic.map(tokenize_and_realign, batched=True)
tokenized_mimic = tokenized_mimic.remove_columns(['tokens', 'tags'])

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

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

#model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", id2label=id2label, label2id=label2id)
#model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner", num_labels=3, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
#model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-base-cased-v1.2", id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=3, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=5e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=2,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="steps",
    save_strategy="steps",
    logging_steps=10,
    eval_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(#LossTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_mimic["train"],
    eval_dataset=tokenized_mimic["validation"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.save_model('model/end/')

trainer.evaluate()

logits, labels, _ = trainer.predict(tokenized_mimic["test"])
predictions = np.argmax(logits, axis=-1)

# Remove ignored index (special tokens)
true_predictions = [
    [labels_bio[p] for (p, l) in zip(prediction, label) if l != -100]#(l != -100 and l != 0)]
    for prediction, label in zip(predictions, labels)
]
true_labels = [
    [labels_bio[l] for l in label if l != -100]#(l != -100 and l != 0)]
    for label in labels
]

results = seqeval.compute(predictions=true_predictions, references=true_labels)
print('Only MIMIC')
print(results)
