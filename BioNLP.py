import pandas as pd
import re

df = pd.read_csv(r'BioNLP2023-1A-Train.csv')
#remove rows with empty GT
df = df.dropna(subset=['Summary']).reset_index(drop=True)
#sustituir por los tags (nombres por Pacient (por ejemplo))
def cleanDeIdentification(x):
    f = re.search(r'\[\*\*(.*?)\*\*\]', str(x))
    while f:
        print('%02d-%02d: %s' % (f.span()[0], f.span()[1], f.group()))
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
        print("After: " + nf)
        x = nf.join([str(x)[:f.span()[0]],str(x)[f.span()[1]:]])
        f = re.search(r'\[\*\*(.*?)\*\*\]', str(x))        
    return x

df[['Assessment','Subjective Sections','Objective Sections']] = df[['Assessment','Subjective Sections','Objective Sections']].applymap(cleanDeIdentification)
#df = df.replace(to_replace ='\[\*\*(.*?)\*\*\]', value = '', regex = True)

#remove measurements
df = df.replace(to_replace ='(\d*(\,|\.)?\d+)\s?(mL|kg|C |cmH2O|%|inch|mmHg|bpm|insp\/min|L\/min|g\/dL|mg\/dL|mEq\/L|mg\/dL|mmol\/L|K\/uL+)', value = '', regex = True)

for idx in range(len(df.index)):
    #remove image id
    df.loc[idx]['Objective Sections'] = re.sub(r'\[?image\d*.jpg\]','',str(df.loc[idx]['Objective Sections']), flags=re.M)
    #remove data number titles
    #df.loc[idx]['Objective Sections'] = re.sub(r'SpO2:|Total In:|PO:|TF:|IVF:|Blooad products:|Total out:|Urine:|NG:|Stool:|Drains:|Balance:|Tmax:|Tcurrent:|WBC|Hct|TCO2|Cr|Plt|Glucose','',str(df.loc[idx]['Objective Sections']), flags=re.M)
    #remove numbers
    #df.loc[idx]['Objective Sections'] = re.sub(r'^\s*(\d*(\,|\.)?\d+\s*)+$','',str(df.loc[idx]['Objective Sections']), flags=re.M)
    #remove extra lines
    df.loc[idx]['Objective Sections'] = "".join([s for s in df.loc[idx]['Objective Sections'].splitlines(True) if s.strip("\n")])

#lowercase
df[['Assessment','Subjective Sections','Objective Sections']] = df[['Assessment','Subjective Sections','Objective Sections']].apply(lambda x: x.str.lower())
#remove change of line
df = df.replace('\n',' ', regex=True)
#remove multiple spaces
df[['Assessment','Subjective Sections','Objective Sections']] = df[['Assessment','Subjective Sections','Objective Sections']].replace(to_replace=' +', value=' ', regex=True)

df.to_csv(r'BioNLP_PP.csv', index=False)
print(df.dtypes)
print('What columns of SOAP to include:\n' + 
        '\t1 - AP\n' +
        '\t2 - SAP\n' + 
        '\t3 - SOAP\n')
option = input()
option = int(option)
#merge columns into one
if option == 1:
    df['Text'] = df['Assessment']
    df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections'])
    df.to_csv(r'BioNLP_PP_AP.csv', index=False)
elif option == 2:
    df['Text'] =  df['Subjective Sections'].astype(str) + ' ' + df['Assessment'].astype(str)
    df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections'])
    df.to_csv(r'BioNLP_PP_SAP.csv', index=False)
elif option == 3:
    df['Text'] = df['Subjective Sections'].astype(str) + ' ' + df['Objective Sections'].astype(str) + ' ' + df['Assessment'].astype(str)
    df = df.drop(columns=['File ID', 'Assessment','Subjective Sections','Objective Sections'])
    df.to_csv(r'BioNLP_PP_SOAP.csv', index=False)    
else:
    print(option)    


print(df)
#print(df.iloc[20]['Objective Sections'])

#active learning: finetuning with other two dataset and then this other dataset in order to make sure the labels are good