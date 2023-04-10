import pandas as pd
from ast import literal_eval

tags = []
tokens = []
summaries = []
words = []

df = pd.read_csv('Preprocessing/NER/BioNLP_NER_man_2_with_Ref.csv')
for i in range(len(df.index)):
    tags.append(literal_eval(df.loc[i]['tags_ap']) + literal_eval(df.loc[i]['tags_s']))# + literal_eval(df.loc[i]['tags_o']))
    tokens.append(literal_eval(df.loc[i]['tokens_ap']) + literal_eval(df.loc[i]['tokens_s']))# + literal_eval(df.loc[i]['tokens_o']))
    summaries.append(df.loc[i]['Summary'])

for i, tag in enumerate(tags):
    w = []
    for j, t in enumerate(tags[i]):
        if t == 1:
            w.append(tokens[i][j])
        elif t == 2:
            w[-1] = w[-1] + ' ' + tokens[i][j]
    words.append(w)

df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections','Summary','tags_ap','tags_s','tags_o','tokens_ap','tokens_s','tokens_o'])
df['words'] = words
df['summary'] = summaries
df.to_csv('Preprocessing/NER/BioT2S2.csv', index=False)
