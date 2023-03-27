import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Sequence, Value
import evaluate
from ast import literal_eval
import torch

from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("model/checkpoint-1560", local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained("model/checkpoint-1560", local_files_only=True)

finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='average')

res = finetunedmodel("acute exacerbation a 59 year-old man presents with afib, malaise, heart attack and hypoxia")
#print(res)

mimic2 = load_dataset('csv', data_files="Preprocessing/NER/BioT2S.csv")
mimic = load_dataset('csv', data_files="Preprocessing/BioNLP_PP_SAP.csv")
mimic['train'] = mimic['train'].add_column('Labels', mimic2['train']['words'])
mimic = mimic['train'].train_test_split(test_size=0.2)
test_valid = mimic['test'].train_test_split(test_size=0.5)
mimic = DatasetDict({
    'train': mimic['train'],
    'test': test_valid['test'],
    'validation': test_valid['train']
})


def fix_words(ex):
    for i, w in enumerate(ex['Text']):
        ex['Labels'][i] = literal_eval(ex['Labels'][i])
        ex['Summary'][i] = str(ex['Summary'][i]).lower()
    return ex

mimic = mimic.map(fix_words, batched=True)
mimic = mimic.cast_column('Labels', Sequence(feature=Value(dtype='string', id=None), length=-1, id=None))
mimic = mimic.cast_column('Text', Value(dtype='string', id=None))
mimic = mimic.cast_column('Summary', Value(dtype='string', id=None))
mimic = mimic.filter(lambda example: len(example["Text"]) > 0)

print(mimic)

rouge = evaluate.load("rouge")
log = open("predictions.log", "w")
for ex in mimic['test']:
    predictions = []
    references = []
    words = ""
    res = finetunedmodel(ex['Text'])
    for match in res:
        words += match['word'] + '; '
    labels = '; '.join(list(dict.fromkeys(ex['Labels'])))
    predictions.append(words)
    references.append(ex['Summary'])
    log.write('Pred= ' + words + '\n')
    log.write('Labels= ' + labels + '\n')
    log.write('Sum=' + ex['Summary'] + '\n')
    log.write('\n')
    rouge.add_batch(predictions=predictions, references=references)

final_score = rouge.compute()

print(final_score)