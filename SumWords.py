import transformers
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from datasets import load_dataset, DatasetDict, Sequence, Value
import evaluate
from ast import literal_eval
import torch
import medialpy
from transformers import AutoModelForTokenClassification, pipeline, AutoTokenizer

#from ray.train.huggingface import HuggingFaceCheckpoint

#checkpoint = HuggingFaceCheckpoint(local_path="model")
#tokenizer = checkpoint.get_tokenizer()
#model = checkpoint.get_model()
tokenizer = AutoTokenizer.from_pretrained("ray_results/_objective_2023-03-30_17-10-23/_objective_fe7d1_00000_0_learning_rate=0.0000,num_train_epochs=3,weight_decay=0.2852_2023-03-30_17-10-23/checkpoint_003000/checkpoint-3000", local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained("ray_results/_objective_2023-03-30_17-10-23/_objective_fe7d1_00000_0_learning_rate=0.0000,num_train_epochs=3,weight_decay=0.2852_2023-03-30_17-10-23/checkpoint_003000/checkpoint-3000", local_files_only=True)

finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='max')

print('Model loaded')

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
log = open("ray_results/abbr/predictions0.log", "w")
for ex in mimic['test']:
    predictions = []
    references = []
    words = ""
    res = finetunedmodel(ex['Text'])
    for match in res:
        words += match['word'] + '; '
        if medialpy.exists(match['word'].upper()):
            term = medialpy.find(match['word'].upper())
            print(match['word'] + ': ')
            print(term.meaning)
            for m in term.meaning:
                words += m + '; '
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
log.write('\n')
log.write('\n')
log.write(str(final_score))