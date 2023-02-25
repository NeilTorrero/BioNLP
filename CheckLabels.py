from ast import literal_eval
import pandas as pd

check = []
df = pd.read_csv(r'BioNLP_NER.csv')
for idx in range(len(df.index)):
    print('\n' + str(idx) + 'Assessment')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_ap'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_ap'])[i]) + ')]')
    print(str(idx) + df.loc[idx]['Summary'])
    print('Needs manual checking?')
    a = input()
    if a == 'y' or a == 'Y':
        check.append(str(idx) + ' - A')
    print('\n' + str(idx) + 'Subjective')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_s'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_s'])[i]) + ')]')
    print(str(idx) + df.loc[idx]['Summary'])
    print('Needs manual checking?')
    a = input()
    if a == 'y' or a == 'Y':
        check.append(str(idx) + ' - S')
    print('\n' + str(idx) + 'Objective')
    for i, word in enumerate(literal_eval(df.loc[idx]['tokens_o'])):
        print(word + '[(' + str(literal_eval(df.loc[idx]['tags_o'])[i]) + ')]')
    print(str(idx) + df.loc[idx]['Summary'])
    print('Needs manual checking?')
    a = input()
    if a == 'y' or a == 'Y':
        check.append(str(idx) + ' - O')

    with open('checklist.txt','w') as file:
	    file.write('\n'.join(check))

print(check)