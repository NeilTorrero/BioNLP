import transformers, torch, evaluate
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer, AutoModelForTokenClassification, pipeline, AutoTokenizer
from datasets import load_dataset, DatasetDict, Sequence, Value
from ast import literal_eval
import medialpy, os, glob


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

for i in range(10):
    for file in glob.glob('ray_results/_objective_2023-04-02_15-13-02/_objective_18c6a_0000' + str(i) + '*/checkpoint_*/checkpoint-*'):
        rouge = evaluate.load("rouge")
        print(file)
        
        tokenizer = AutoTokenizer.from_pretrained(file, local_files_only=True)
        model = AutoModelForTokenClassification.from_pretrained(file, local_files_only=True)

        finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='average')

        print('Model loaded')

        log = open("ray_results/abbr/predictions" + str(i) + ".log", "w")
        log.write('Normal')
        for ex in mimic['test']:
            predictions = []
            references = []
            words = ""
            res = finetunedmodel(ex['Text'])
            for match in res:
                words += match['word'] + ' ; '
            labels = ' ; '.join(list(dict.fromkeys(ex['Labels'])))
            predictions.append(words)
            references.append(ex['Summary'])
            log.write('Pred= ' + words + '\n')
            log.write('Labels= ' + labels + '\n')
            log.write('Sum=' + ex['Summary'] + '\n')
            log.write('\n')
            rouge.add_batch(predictions=predictions, references=references)

        final_score = rouge.compute()

        print(final_score)
        log.write('\n\n')
        log.write(str(final_score))
        
        rouge = evaluate.load("rouge")
        log.write('\n\n\n')
        log.write('Abbrebiations')
        for ex in mimic['test']:
            predictions = []
            references = []
            words = ""
            res = finetunedmodel(ex['Text'])
            for match in res:
                words += match['word'] + ' ; '
                if medialpy.exists(match['word'].upper()):
                    term = medialpy.find(match['word'].upper())
                    print(match['word'] + ': ')
                    print(term.meaning)
                    if len(term.meaning) == 1:
                        words += term.meaning[0] + ' ; '
                    #for m in term.meaning:
                    #    words += m + ' ; '
            labels = ' ; '.join(list(dict.fromkeys(ex['Labels'])))
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
