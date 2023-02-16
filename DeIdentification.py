import pandas as pd
import re

df = pd.read_csv(r'BioNLP2023-1A-Train.csv')
occurences = set()
new_occ = set()
#sustituir por los tags (nombres por Pacient (por ejemplo))
for idx in range(len(df.index)):
    # check for special case Age Over #Age #Number
    for f in re.findall(r'\[\*\*(.*?)\*\*\]',str(df.loc[idx]['Assessment']), flags=re.M):
        occurences.add(f)
        nf = re.sub(r'\((.*?)\)','',f, flags=re.M)
        if re.search(r'[a-zA-Z]+', nf):
            nf = re.sub(r'[0-9]', '', nf, flags=re.M)
        else:
            nf = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'Date',  nf, flags=re.M)
            nf = re.sub(r'\d{1,2}-\d{1,2}', 'Time',  nf, flags=re.M)
            nf = re.sub(r'(\d{2}|\d{1}-)\/\d{4}', 'Place?',  nf, flags=re.M)
            nf = re.sub(r'\d{4}', 'Location?',  nf, flags=re.M)
            nf = re.sub(r'\d{2}', 'Number',  nf, flags=re.M)
        new_occ.add(nf.strip())
    for f in re.findall(r'\[\*\*(.*?)\*\*\]',str(df.loc[idx]['Subjective Sections']), flags=re.M):
        occurences.add(f)
        nf = re.sub(r'\((.*?)\)','',f, flags=re.M)
        if re.search(r'[a-zA-Z]+', nf):
            nf = re.sub(r'[0-9]', '', nf, flags=re.M)
        else:
            nf = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'Date',  nf, flags=re.M)
            nf = re.sub(r'\d{1,2}-\d{1,2}', 'Time',  nf, flags=re.M)
            nf = re.sub(r'(\d{2}|\d{1}-)\/\d{4}', 'Place?',  nf, flags=re.M)
            nf = re.sub(r'\d{4}', 'Location?',  nf, flags=re.M)
            nf = re.sub(r'\d{2}', 'Number',  nf, flags=re.M)
        new_occ.add(nf.strip())
    for f in re.findall(r'\[\*\*(.*?)\*\*\]',str(df.loc[idx]['Objective Sections']), flags=re.M):
        occurences.add(f)
        nf = re.sub(r'\((.*?)\)','',f, flags=re.M)
        if re.search(r'[a-zA-Z]+', nf):
            nf = re.sub(r'[0-9]', '', nf, flags=re.M)
        else:
            nf = re.sub(r'\d{4}-\d{1,2}-\d{1,2}', 'Date',  nf, flags=re.M)
            nf = re.sub(r'\d{1,2}-\d{1,2}', 'Time',  nf, flags=re.M)
            nf = re.sub(r'(\d{2}|\d{1}-)\/\d{4}', 'Place?',  nf, flags=re.M)
            nf = re.sub(r'\d{4}', 'Location?',  nf, flags=re.M)
            nf = re.sub(r'\d{2}', 'Number',  nf, flags=re.M)
        new_occ.add(nf.strip())

print('\n\n---------Occurences-------\n')
print(occurences)
print('\n\n---------Fixes-----------\n')
print(new_occ)