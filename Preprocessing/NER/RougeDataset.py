#
# Evaluating dataset for Word to Summary to Rouge
#
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

mimic = load_dataset('csv', data_files="Preprocessing/NER/Resources/BioT2S2.csv")


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

from rouge_score import rouge_scorer

def applyPyRouge(gts, preds):
    scorer = rouge_scorer.RougeScorer(['rouge1','rouge2','rougeL'])
    scores = scorer.score(gts, preds) 
    return scores 

    
def compute_rouge(gt, pred):
    sc = applyPyRouge(gt, pred)
    return sc['rougeL'].precision, sc['rougeL'].recall, sc['rougeL'].fmeasure

import difflib
import re, numpy as np

metrics = {}
precs, recs, f1s = [], [], [] 
log = open("Preprocessing/NER/Resources/dataset_rouge.log", "w")
for idx, ex in enumerate(mimic['train']):
    predictions = []
    references = []
    rouge2 = evaluate.load("rouge")
    #wlist = ex['words']
    wlist = list(dict.fromkeys(ex['words']))
    for i, e in enumerate(wlist):
        # fix spaces on special characters (s / p)
        wlist[i] = wlist[i].replace(" / ", "/")
        wlist[i] = wlist[i].replace(" - ", "-")
        wlist[i] = wlist[i].replace(" .", ".")
        for j, ea in enumerate(wlist):
            if i != j:
                if (ea in e) and len(difflib.get_close_matches(ea, [e])) == 1:
                    wlist.pop(j)
    # things like "pea arrest" and "pea" are not merged
    pred = '; '.join(wlist)
    log.write('\n' + str(idx) + '\n')
    log.write('Prediction:\n\t' + pred + '\n')
    log.write('Summary:\n\t' + ex['summary'] + '\n')
    log.write(str(rouge2.compute(predictions=[pred], references=[ex['summary']])))
    log.write(str(compute_rouge(ex['summary'], pred)))
    log.write('\n')
    predictions.append(pred)
    references.append(ex['summary'])

    precision, recall, f1 = compute_rouge(ex['summary'], pred)
    precs.append(precision)
    recs.append(recall)
    f1s.append(f1)
    rouge.add_batch(predictions=predictions, references=references)

final_score = rouge.compute()

log.write('\n\nFinal score\n')
log.write(str(final_score))
print(final_score)

metrics["precision"] = np.mean(precs)
metrics["recall"] = np.mean(recs)
metrics["f1"] = np.mean(f1s)
print(metrics)
log.write('\n')
log.write(str(metrics))