import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, load_dataset, concatenate_datasets

bc5cdr = load_dataset("tner/bc5cdr")
ncbi = load_dataset("ncbi_disease")

bc5cdr = bc5cdr.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))

ncbi = ncbi.remove_columns('id')
ncbi = ncbi.rename_column("ner_tags", "tags")
ncbi = ncbi.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))
#Adapt tags to bc5cdr (1 -> 2, 2 -> 3)
def change_tags(ex):
    for i, tags in enumerate(ex['tags']):
        for j, tag in enumerate(tags):
            if tag == 1:
                ex["tags"][i][j] = 2
            else:
                if tag == 2:
                    ex["tags"][i][j] = 3
    return ex

ncbi_adapted = ncbi.map(change_tags, batched=True)
#data_files = {'train': 'NER_data/BC5CDR/train.json', 'validation': 'NER_data/BC5CDR/valid.json', 'test': 'NER_data/BC5CDR/test.json'}
#bc5cdr = load.load_dataset('json', data_files=data_files)
datasets = DatasetDict()
datasets['train'] = concatenate_datasets([bc5cdr['train'],ncbi_adapted['train']])
datasets['validation'] = concatenate_datasets([bc5cdr['validation'],ncbi_adapted['validation']])
datasets['test'] = concatenate_datasets([bc5cdr['test'],ncbi_adapted['test']])
print(datasets)

labels_bio = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
example = datasets['train'][0]
tokenized = tokenizer(example["tokens"], is_split_into_words=True)
tokens = tokenizer.convert_ids_to_tokens(tokenized["input_ids"])
print(tokens)

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
            elif word_idx != previous_word_idx:  # Only label the first token of a given word.
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_ex["labels"] = labels
    return tokenized_ex

tokenized_dataset = datasets.map(tokenize_and_realign, batched=True)

from transformers import DataCollatorForTokenClassification
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

seqeval = evaluate.load("seqeval")

labels = [labels_bio[i] for i in example["tags"]]

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [labels_bio[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [labels_bio[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

id2label = {0:"O", 1:"B-Chemical", 2:"B-Disease", 3:"I-Disease", 4:"I-Chemical"}
label2id = {"O":0, "B-Chemical":1, "B-Disease":2, "I-Disease":3, "I-Chemical":4}


model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", id2label=id2label, label2id=label2id)
training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

from transformers import pipeline

finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer)

finetunedmodel()