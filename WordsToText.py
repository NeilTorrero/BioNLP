import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Sequence, Value
import evaluate
from ast import literal_eval
import torch

#from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer

#tokenizer = AutoTokenizer.from_pretrained("model/checkpoint-1560", local_files_only=True)
#model = AutoModelForTokenClassification.from_pretrained("model/checkpoint-1560", local_files_only=True)

#finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='average')

#res = finetunedmodel("acute exacerbation a 59 year-old man presents with afib, malaise, heart attack and hypoxia")
#print(res)

mimic = load_dataset('csv', data_files="Preprocessing/NER/BioT2S.csv")
mimic = mimic['train'].train_test_split(test_size=0.2)
test_valid = mimic['test'].train_test_split(test_size=0.5)
mimic = DatasetDict({
    'train': mimic['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})


def fix_words(ex):
    for i, w in enumerate(ex['words']):
        ex['words'][i] = literal_eval(ex['words'][i])
    return ex

mimic = mimic.map(fix_words, batched=True)
mimic = mimic.rename_column("words", "input_ids")
mimic = mimic.rename_column("summary", "labels")
mimic = mimic.cast_column('input_ids', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('labels', Value(dtype='string', id=None))
mimic = mimic.filter(lambda example: len(example["input_ids"]) > 0)

print(mimic)


rouge = evaluate.load("rouge")

def compute_metrics(p):
    predictions, references = p
    print(p)
    results = rouge.compute(predictions=predictions, references=references)
    return {
        "rouge1": results["rouge1"],
        "rouge2": results["rouge2"],
        "rougeL": results["rougeL"],
        "rougeLsum": results["rougelsum"],
    }

tokenizer = T5Tokenizer.from_pretrained("t5-base")

def tokenize(ex):
    for i, label in enumerate(ex["input_ids"]):
        ex["input_ids"][i] = ' '.join(label)

    input_encodings = tokenizer.batch_encode_plus(ex["input_ids"], return_tensors="pt", padding='max_length', max_length=1024)
    target_encodings = tokenizer.batch_encode_plus(ex['labels'], return_tensors="pt", padding='max_length', max_length=1024)
    tokenized_ex = {
        'input_ids': input_encodings['input_ids'], 
        'attention_mask': input_encodings['attention_mask'],
        'target_ids': target_encodings['input_ids'],
        'target_attention_mask': target_encodings['attention_mask']
    }
    return tokenized_ex

tokenized_dataset = mimic.map(tokenize, batched=True)
columns = ['input_ids', 'target_ids', 'attention_mask', 'target_attention_mask']
tokenized_dataset.set_format(type='torch', columns=columns)

model = T5ForConditionalGeneration.from_pretrained("t5-base")


from dataclasses import dataclass
from typing import Dict, List, Optional

from transformers import (
    DataCollator,
    Trainer,
    TrainingArguments,
)

@dataclass
class T2TDataCollator(DataCollator):
    def collate_batch(self, batch: List) -> Dict[str, torch.Tensor]:
        input_ids = torch.stack([example['input_ids'] for example in batch])
        lm_labels = torch.stack([example['target_ids'] for example in batch])
        lm_labels[lm_labels[:, :] == 0] = -100
        attention_mask = torch.stack([example['attention_mask'] for example in batch])
        decoder_attention_mask = torch.stack([example['target_attention_mask'] for example in batch])
        
        return {
            'input_ids': input_ids, 
            'attention_mask': attention_mask,
            'lm_labels': lm_labels, 
            'decoder_attention_mask': decoder_attention_mask
        }

data_collator = T2TDataCollator()

training_args = TrainingArguments(
    output_dir="model_w2t",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    num_train_epochs=5,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="loss"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=mimic["train"],
    eval_dataset=mimic["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()