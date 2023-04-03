import transformers
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, TrainingArguments, Trainer, Seq2SeqTrainer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq
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
mimic = mimic.cast_column('words', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('summary', Value(dtype='string', id=None))
mimic = mimic.filter(lambda example: len(example["words"]) > 0)

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

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")

prefix = "summarize: "

def tokenize(ex):
    for i, label in enumerate(ex["words"]):
        ex["words"][i] = ' | '.join(label)

    inputs = [prefix + ex for ex in ex["words"]]
    model_inputs = tokenizer(inputs, return_tensors="pt", padding='max_length', max_length=1024)
    
    labels = tokenizer(text_target=ex['summary'], return_tensors="pt", padding='max_length', max_length=1024)
    
    model_inputs["labels"] = labels['input_ids']
    return model_inputs

tokenized_dataset = mimic.map(tokenize, batched=True)
tokenized_dataset = tokenized_dataset.remove_columns(['words', 'summary'])
print(tokenized_dataset)

model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model,label_pad_token_id=-100)

training_args = Seq2SeqTrainingArguments(
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

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.evaluate()