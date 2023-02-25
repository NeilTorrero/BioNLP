from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import pandas as pd
import re

entity = {
    "0": 0,
    "B-DISEASE": 2,
    "I-DISEASE": 3,
}
tokenizer = AutoTokenizer.from_pretrained("alvaroalon2/biobert_diseases_ner")
#tokenizer = AutoTokenizer.from_pretrained("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")

model = AutoModelForTokenClassification.from_pretrained("alvaroalon2/biobert_diseases_ner")

ner = pipeline("ner", model=model, tokenizer=tokenizer)

df = pd.read_csv(r'BioNLP_PP.csv')

tokens_ap = []
tags_ap = []
tokens_s = []
tags_s = []
tokens_o = []
tags_o = []
for idx in range(len(df.index)):
    print(str(idx) + '/' + str(len(df.index)-1))
    text_ap = df.loc[idx]['Assessment']
    if not pd.isna(text_ap):
        ner_text_ap = ner(text_ap)
        tokens = []
        tags = []
        for word in ner_text_ap:
            tokens.append(str(word['word']))
            tags.append(int(entity[word['entity']]))
        for i, token in reversed(list(enumerate(tokens))):
            #print(i, token, tags[i])
            if "##" in token:
                tokens[i-1] += tokens[i][2:]
                tokens.pop(i)
                tags.pop(i)
        tokens_ap.append(list(tokens))
        tags_ap.append(list(tags))
    else:
        tokens_ap.append([])
        tags_ap.append([])
    text_s = df.loc[idx]['Subjective Sections']
    if not pd.isna(text_s):
        ner_text_s = ner(text_s)
        tokens = []
        tags = []
        for word in ner_text_s:
            tokens.append(str(word['word']))
            tags.append(int(entity[word['entity']]))
        for i, token in reversed(list(enumerate(tokens))):
            #print(i, token, tags[i])
            if "##" in token:
                tokens[i-1] += tokens[i][2:]
                tokens.pop(i)
                tags.pop(i)
        tokens_s.append(tokens)
        tags_s.append(list(tags))
    else:
        tokens_s.append([])
        tags_s.append([])
    text_o = df.loc[idx]['Objective Sections']
    if not pd.isna(text_o):
        ner_text_o = ner(text_o)
        tokens = []
        tags = []
        for word in ner_text_o:
            tokens.append(str(word['word']))
            tags.append(int(entity[word['entity']]))
        for i, token in reversed(list(enumerate(tokens))):
            #print(i, token, tags[i])
            if "##" in token:
                tokens[i-1] += tokens[i][2:]
                tokens.pop(i)
                tags.pop(i)
        tokens_o.append(list(tokens))
        tags_o.append(list(tags))
    else:
        tokens_o.append([])
        tags_o.append([])


df['tokens_ap'] = tokens_ap.astype('object')
df['tags_ap'] = tags_ap.astype('object')
df['tokens_s'] = tokens_s.astype('object')
df['tags_s'] = tags_s.astype('object')
df['tokens_o'] = tokens_o.astype('object')
df['tags_o'] = tags_o.astype('object')
df.to_csv('BioNLP_NER.csv')

