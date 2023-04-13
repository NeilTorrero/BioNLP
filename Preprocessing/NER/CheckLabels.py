#
# Script for manual review of NER labeling
#
from ast import literal_eval
import pandas as pd

check = []
df = pd.read_csv(r'Resources/BioNLP_NER.csv')
jump = 0
for idx in range(len(df.index)):
    if jump == 0:
        print('Jump to:')
        idx = int(input())
        jump = 1
    print('\n' + str(idx) + ' Assessment')
    print(df.loc[idx]['Assessment'])
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_ap'])):
        if literal_eval(df.loc[idx]['tags_ap'])[i] != 0:
            print(word + ' [' + str(literal_eval(df.loc[idx]['tags_ap'])[i]) + ']')
    print(str(idx)  + ' ' + df.loc[idx]['Summary'])
    print('\n' + str(idx) + ' Assessment')
    print('Needs manual checking?')
    a = input()
    if a != '':
        check.append(str(idx) + ' - A' + ' # ' + a)
    print('\n' + str(idx) + ' Subjective')
    print(df.loc[idx]['Subjective Sections'])
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_s'])):
        if literal_eval(df.loc[idx]['tags_s'])[i] != 0:
            print(word + ' [' + str(literal_eval(df.loc[idx]['tags_s'])[i]) + ']')
    print(str(idx) + ' ' + df.loc[idx]['Summary'])
    print('\n' + str(idx) + ' Subjective')
    print('Needs manual checking?')
    a = input()
    if a != '':
        check.append(str(idx) + ' - S' + ' # ' + a)
    print('\n' + str(idx) + ' Objective')
    print(df.loc[idx]['Objective Sections'])
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_o'])):
        if literal_eval(df.loc[idx]['tags_o'])[i] != 0:
            print(word + ' [' + str(literal_eval(df.loc[idx]['tags_o'])[i]) + ']')
    print(str(idx) + ' ' + df.loc[idx]['Summary'])
    print('\n' + str(idx) + ' Objective')
    print('Needs manual checking?')
    a = input()
    if a != '':
        check.append(str(idx) + ' - O' + ' # ' + a)

    with open('Resources/checklist.txt','w') as file:
	    file.write('\n'.join(check))

print(check)