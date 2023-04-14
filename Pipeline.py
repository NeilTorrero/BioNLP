#
# Script of pipeline, from text to final summary prediction
#
import transformers, evaluate, re, os
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from datasets import load_dataset, Value
import pandas as pd

df = pd.read_csv(r'Resources/BioNLP2023-1A-Train.csv')
#remove rows with empty GT
df = df.dropna(subset=['Summary']).reset_index(drop=True)
#sustituir por los tags (nombres por Pacient (por ejemplo))
def cleanDeIdentification(x):
    f = re.search(r'\[\*\*(.*?)\*\*\]', str(x))
    while f:
        nf = f.group()[3:-3]
        nf = re.sub(r'\((.*?)\)','',nf, flags=re.M)
        if re.search(r'[a-zA-Z]+', nf):
            nf = re.sub(r'[0-9]', '', nf, flags=re.M)
        else:
            nf = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'Date', nf, flags=re.M)
            nf = re.sub(r'\d{1,2}-\d{1,2}', 'Time', nf, flags=re.M)
            nf = re.sub(r'(\d{2}|\d{1}-)\/\d{4}', 'Place?', nf, flags=re.M)
            nf = re.sub(r'\d{4}', 'Location?', nf, flags=re.M)
            nf = re.sub(r'\d{2}', 'Number', nf, flags=re.M)
        nf = nf.strip()
        x = nf.join([str(x)[:f.span()[0]],str(x)[f.span()[1]:]])
        f = re.search(r'\[\*\*(.*?)\*\*\]', str(x))        
    return x

df[['Assessment','Subjective Sections']] = df[['Assessment','Subjective Sections']].applymap(cleanDeIdentification)
#df = df.replace(to_replace ='\[\*\*(.*?)\*\*\]', value = '', regex = True)

#remove measurements
df = df.replace(to_replace ='(\d*(\,|\.)?\d+)\s?(mL|kg|C |cmH2O|%|inch|mmHg|bpm|insp\/min|L\/min|g\/dL|mg\/dL|mEq\/L|mg\/dL|mmol\/L|K\/uL+)', value = '', regex = True)

#lowercase
df[['Assessment','Subjective Sections']] = df[['Assessment','Subjective Sections']].apply(lambda x: x.str.lower())
#remove change of line
df = df.replace('\n',' ', regex=True)
#remove multiple spaces
df[['Assessment','Subjective Sections']] = df[['Assessment','Subjective Sections']].replace(to_replace=' +', value=' ', regex=True)


df['Text'] =  df['Assessment'].astype(str) + ' ' + df['Subjective Sections'].astype(str)
df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections'])

df.to_csv(r'Resources/Dataset_Test.csv', index=False)



mimic = load_dataset('csv', data_files="Resources/Dataset_Test.csv")
mimic = mimic.cast_column('Text', Value(dtype='string', id=None))
mimic = mimic.cast_column('Summary', Value(dtype='string', id=None))
mimic = mimic.filter(lambda example: len(example["Text"]) > 0)



rouge = evaluate.load("rouge")
file = 'model/end/'
tokenizer = AutoTokenizer.from_pretrained(file, local_files_only=True)
model = AutoModelForTokenClassification.from_pretrained(file, local_files_only=True)

finetunedmodel = pipeline("ner", model=model, tokenizer=tokenizer, aggregation_strategy='average')

print('Model loaded')

file = open('Resources/system.txt', 'w')
for ex in mimic['train']:
    predictions = []
    references = []
    words = ""
    res = finetunedmodel(ex['Text'])
    for match in res:
        words += match['word'] + '; '
    predictions.append(words)
    file.write(words + '\n')
    references.append(ex['Summary'])
    rouge.add_batch(predictions=predictions, references=references)

final_score = rouge.compute()

print(final_score)
os.remove(r'Resources/Dataset_Test.csv')