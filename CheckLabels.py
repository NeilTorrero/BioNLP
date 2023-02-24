from ast import literal_eval
import pandas as pd

df = pd.read_csv(r'BioNLP_NER.csv')
for idx in range(len(df.index)):
    print('\nAssessment')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_ap'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_ap'])[i]) + ')]')
    print(df.loc[idx]['Summary'])
    input()
    print('\nSubjective')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_s'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_s'])[i]) + ')]')
    print(df.loc[idx]['Summary'])
    input()
    print('\nObjective')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_o'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_o'])[i]) + ')]')
    print(df.loc[idx]['Summary'])
    input()