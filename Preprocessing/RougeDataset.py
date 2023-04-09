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

mimic = load_dataset('csv', data_files="Preprocessing/NER/BioT2S2.csv")
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
        ex['summary'][i] = str(ex['summary'][i]).lower()
    return ex

mimic = mimic.map(fix_words, batched=True)
mimic = mimic.cast_column('words', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('summary', Value(dtype='string', id=None))
mimic = mimic.filter(lambda example: len(example["words"]) > 0)

print(mimic)

rouge = evaluate.load("rouge")
import difflib
for ex in mimic['train']:
    predictions = []
    references = []
    wlist = list(dict.fromkeys(ex['words']))
    for i, e in enumerate(wlist):
        for j, ea in enumerate(wlist):
            if i != j:
                if (ea in e) and len(difflib.get_close_matches(ea, [e])) == 1:
                    wlist.pop(j)
    predictions.append('; '.join(wlist))
    references.append(ex['summary'])
    rouge.add_batch(predictions=predictions, references=references)

final_score = rouge.compute()

print(final_score)