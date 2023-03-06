import transformers
from transformers import AutoTokenizer, AutoModel, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np
from ast import literal_eval
from datasets import Dataset, DatasetDict, ClassLabel, Sequence, Value, load_dataset, concatenate_datasets

bc5cdr = load_dataset("tner/bc5cdr")
ncbi = load_dataset("ncbi_disease")
mimic = load_dataset('csv', data_files="BioNLP_dataset.csv")
mimic = mimic['train'].train_test_split(test_size=0.2)

# Adapt BC5CDR
bc5cdr = bc5cdr.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))
bc5cdr = bc5cdr.filter(lambda example: len(example["tags"]) > 0)

# Adapt NCBI
ncbi = ncbi.filter(lambda example: len(example["ner_tags"]) > 0)
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

# Adapt MIMIC don't needed here already been done in preprocessing
def int_tags(ex):
    ex_i = []
    for i, tags in enumerate(ex['tags']):
        ex['tags'][i] = literal_eval(ex['tags'][i])
        ex['tokens'][i] = literal_eval(ex['tokens'][i])
    return ex

mimic = mimic.map(int_tags,batched=True)
mimic = mimic.cast_column('tokens', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('tags', Sequence(feature=ClassLabel(names=["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"], id=None), length=-1, id=None))
mimic = mimic.filter(lambda example: len(example["tags"]) > 0)


# Merge
datasets = DatasetDict()
datasets['train'] = concatenate_datasets([bc5cdr['train'],ncbi_adapted['train'],mimic['train']])
datasets['validation'] = concatenate_datasets([bc5cdr['validation'],ncbi_adapted['validation']])
datasets['test'] = concatenate_datasets([bc5cdr['test'],ncbi_adapted['test'],mimic['test']])
print(datasets)

labels_bio = ["O", "B-Chemical", "B-Disease", "I-Disease", "I-Chemical"]

# Tokenize and adapt datasets to tokenization
#tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
#tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
example = datasets['train'][0]
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

#model = AutoModelForTokenClassification.from_pretrained("dmis-lab/biobert-v1.1", id2label=id2label, label2id=label2id)
#model = AutoModelForTokenClassification.from_pretrained("emilyalsentzer/Bio_ClinicalBERT", id2label=id2label, label2id=label2id)
model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner", id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)

training_args = TrainingArguments(
    output_dir="model",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
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

res = finetunedmodel("ultrasound - at date 11:32 am - md first name last name name but coulnd't leave voicemail because her voicemail wasn't set up - patient refused ngt for kayexalate; k 6.5 at noon; ecg unremarkable. was able to take po well in p.m., received kayexalate; k down to 5.0 - renal u/s: no hydronephrosis no known drug allergies changes to and f review of systems is unchanged from admission except as noted below review of systems: last dose of antibiotics: azithromycin - date 10:46 am infusions: other icu medications: furosemide (lasix) - date 02:45 pm heparin sodium (prophylaxis) - date 04:23 pm other medications: flowsheet data as of date 08:14 am vital signs hemodynamic monitoring fluid balance 24 hours since number am tmax: (98 tcurrent: (97.8 hr: 72 (54 - 72) bpm bp: 120/59(74) {90/49(60) - 162/128(136)} mmhg rr: 20 (12 - 25) insp/min spo2: heart rhythm: sr (sinus rhythm) wgt (current): (admission): total in: po: tf: ivf: blood products: total out: urine: ng: stool: drains: balance: - respiratory support o2 delivery device: cpap mask ventilator mode: cpap/psv vt (spontaneous): 340 (340 - 340) ml ps : rr (spontaneous): 18 peep: fio2: pip: spo2: abg: 7.29/61/84.numeric identifier/30/0 ve: pao2 / fio2: 168 peripheral vascular: (right radial pulse: not assessed), (left radial pulse: not assessed), (right dp pulse: not assessed), (left dp pulse: not assessed) skin: not assessed neurologic: responds to: not assessed, movement: not assessed, tone: not assessed / date 01:50 am date 05:25 am date 07:42 am date 09:06 am date 06:10 pm date 05:14 am wbc 12.6 12.6 hct 40.2 37.6 plt 251 254 cr 2.5 2.5 2.1 tco2 30 33 31 glucose telephone/fax other labs: ck / ckmb / troponin-t:44//, differential-neuts:, lymph:, mono:, eos:, lactic acid:, ca++:, mg++:, po4: h/o hyperkalemia (high potassium, hyperpotassemia) .h/o hyperglycemia chronic obstructive pulmonary disease (copd, bronchitis, emphysema) with acute exacerbation a 59 year-old man presents with malaise and hypoxia")
print(res)