import pandas as pd
from ast import literal_eval
import re

# Options: 
#   1- Merge 3 columns together
#   2- Put every column as diferent examples
#   3- The 2 previous ones but excluding the Objective column
df = pd.read_csv(r'Preprocessing/NER/BioNLP_NER_man.csv')
tags = []
tokens = []
print('Separate sections in different examples?')
separate = int(input())
print('Skip Objective section?')
skip = int(input())

for i in range(len(df.index)):
    if separate == 0:
        if skip == 0:
            tag = literal_eval(df.loc[i]['tags_ap']) + literal_eval(df.loc[i]['tags_s']) + literal_eval(df.loc[i]['tags_o'])
            tags.append(tag)
            token = literal_eval(df.loc[i]['tokens_ap']) + literal_eval(df.loc[i]['tokens_s']) + literal_eval(df.loc[i]['tokens_o'])
            tokens.append(token)
        else:
            tag = literal_eval(df.loc[i]['tags_ap']) + literal_eval(df.loc[i]['tags_s'])
            tags.append(tag)
            token = literal_eval(df.loc[i]['tokens_ap']) + literal_eval(df.loc[i]['tokens_s'])
            tokens.append(token)
    else:
        if skip == 0:
            tags.append(literal_eval(df.loc[i]['tags_ap']))
            tokens.append(literal_eval(df.loc[i]['tokens_ap']))
            tags.append(literal_eval(df.loc[i]['tags_s']))
            tokens.append(literal_eval(df.loc[i]['tokens_s']))
            tags.append(literal_eval(df.loc[i]['tags_o']))
            tokens.append(literal_eval(df.loc[i]['tokens_o']))
        else:
            tags.append(literal_eval(df.loc[i]['tags_ap']))
            tokens.append(literal_eval(df.loc[i]['tokens_ap']))
            tags.append(literal_eval(df.loc[i]['tags_s']))
            tokens.append(literal_eval(df.loc[i]['tokens_s']))
    #print(tag)
    #print(token)
    #input()

if separate == 0:
    df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections','Summary','tags_ap','tags_s','tags_o','tokens_ap','tokens_s','tokens_o'])
    df['tokens'] = tokens
    df['tags'] = tags
    df.to_csv(r'BioNLP_dataset.csv', index=False)
else:
    ndf = pd.DataFrame({'tokens': tokens, 'tags': tags})
    ndf.to_csv(r'BioNLP_dataset2.csv', index=False)